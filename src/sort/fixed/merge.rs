//! K-way merge for fixed-size records.
//!
//! Only keys live in the loser tree. A winning record's value is read from its
//! source straight into a stack buffer and handed to the sink, so no heap
//! allocation happens per record.

use crate::diskio::io_stats::IoStatsTracker;
use crate::ovc::offset_value_coding_64::{SentinelValue, Sentineled};
use crate::ovc::tree_of_losers::LoserTree;
use std::os::unix::fs::FileExt;

/// A sorted stream of fixed-size records that is read key first.
pub trait FixedMergeSource<const K: usize, const V: usize> {
    /// Read the next key. On `true` the value must be consumed next with
    /// `read_value` or `skip_value`.
    fn next_key(&mut self, key: &mut [u8; K]) -> bool;
    fn read_value(&mut self, value: &mut [u8; V]);
    fn skip_value(&mut self);
}

/// Destination for merged records.
pub trait FixedSink<const K: usize, const V: usize> {
    fn put(&mut self, key: &[u8; K], value: &[u8; V]);
}

/// Loser-tree element: compares keys byte by byte (lexicographically).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedKey<const K: usize>(pub [u8; K]);

/// Merge any number of sources into `sink`. Returns the number of records.
pub fn merge_fixed<S, W, const K: usize, const V: usize>(sources: Vec<S>, sink: &mut W) -> usize
where
    S: FixedMergeSource<K, V>,
    W: FixedSink<K, V>,
{
    let mut value = [0u8; V];
    match sources.len() {
        0 => 0,
        1 => {
            let mut source = sources.into_iter().next().unwrap();
            let mut key = [0u8; K];
            let mut count = 0;
            while source.next_key(&mut key) {
                source.read_value(&mut value);
                sink.put(&key, &value);
                count += 1;
            }
            count
        }
        _ => {
            let mut merge = ZeroCopyMergeFixed::new(sources);
            let mut count = 0;
            while let Some((key, source)) = merge.peek() {
                merge.sources[source].read_value(&mut value);
                sink.put(&key, &value);
                merge.advance(source);
                count += 1;
            }
            count
        }
    }
}

/// Consume every record from `sources` in merge order without writing it.
pub fn discard_fixed<S, const K: usize, const V: usize>(sources: Vec<S>) -> usize
where
    S: FixedMergeSource<K, V>,
{
    struct Null;
    impl<const K: usize, const V: usize> FixedSink<K, V> for Null {
        #[inline]
        fn put(&mut self, _key: &[u8; K], _value: &[u8; V]) {}
    }
    merge_fixed(sources, &mut Null)
}

struct ZeroCopyMergeFixed<S, const K: usize, const V: usize> {
    tree: LoserTree<Sentineled<FixedKey<K>>>,
    sources: Vec<S>,
}

impl<S: FixedMergeSource<K, V>, const K: usize, const V: usize> ZeroCopyMergeFixed<S, K, V> {
    fn new(mut sources: Vec<S>) -> Self {
        let initial = sources
            .iter_mut()
            .map(|source| {
                let mut key = [0u8; K];
                if source.next_key(&mut key) {
                    Sentineled::new(FixedKey(key))
                } else {
                    Sentineled::late_fence()
                }
            })
            .collect();
        Self {
            tree: LoserTree::new(initial),
            sources,
        }
    }

    /// Current winner's key and source index, or `None` once all are exhausted.
    #[inline]
    fn peek(&self) -> Option<([u8; K], usize)> {
        match self.tree.peek()? {
            (Sentineled::Normal(key), source) => Some((key.0, source)),
            _ => None,
        }
    }

    /// Replace the winner (from `source`) with that source's next key.
    #[inline]
    fn advance(&mut self, source: usize) {
        let mut key = [0u8; K];
        if self.sources[source].next_key(&mut key) {
            self.tree.push(Sentineled::new(FixedKey(key)));
        } else {
            self.tree.mark_current_exhausted();
        }
    }
}

/// Size of each packed-output write. Kept per merge thread.
const PACKED_WRITE_BUFFER_BYTES: usize = 4 << 20;

/// Writes one merge thread's records into the shared packed output file,
/// starting at a fixed byte offset, with positional writes (`pwrite`).
pub struct PackedWriter<'a> {
    file: &'a std::fs::File,
    next_offset: u64,
    buffer: Vec<u8>,
    written: u64,
    io_tracker: Option<IoStatsTracker>,
    error: Option<std::io::Error>,
}

impl<'a> PackedWriter<'a> {
    pub fn new(
        file: &'a std::fs::File,
        start_offset: u64,
        io_tracker: Option<IoStatsTracker>,
    ) -> Self {
        Self {
            file,
            next_offset: start_offset,
            buffer: Vec::with_capacity(PACKED_WRITE_BUFFER_BYTES),
            written: 0,
            io_tracker,
            error: None,
        }
    }

    fn flush(&mut self) {
        if self.buffer.is_empty() || self.error.is_some() {
            self.buffer.clear();
            return;
        }
        match self.file.write_all_at(&self.buffer, self.next_offset) {
            Ok(()) => {
                let len = self.buffer.len() as u64;
                if let Some(tracker) = &self.io_tracker {
                    tracker.add_write(1, len);
                }
                self.next_offset += len;
                self.written += len;
            }
            Err(e) => self.error = Some(e),
        }
        self.buffer.clear();
    }

    /// Flush the remaining bytes and return the total number written.
    pub fn finish(mut self) -> Result<u64, String> {
        self.flush();
        match self.error.take() {
            Some(e) => Err(format!("Failed to write packed output: {}", e)),
            None => Ok(self.written),
        }
    }
}

impl<const K: usize, const V: usize> FixedSink<K, V> for PackedWriter<'_> {
    #[inline]
    fn put(&mut self, key: &[u8; K], value: &[u8; V]) {
        self.buffer.extend_from_slice(key);
        self.buffer.extend_from_slice(value);
        if self.buffer.len() + K + V > PACKED_WRITE_BUFFER_BYTES {
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct VecSource {
        records: Vec<([u8; 2], [u8; 1])>,
        pos: usize,
    }

    impl FixedMergeSource<2, 1> for VecSource {
        fn next_key(&mut self, key: &mut [u8; 2]) -> bool {
            if self.pos >= self.records.len() {
                return false;
            }
            *key = self.records[self.pos].0;
            true
        }
        fn read_value(&mut self, value: &mut [u8; 1]) {
            *value = self.records[self.pos].1;
            self.pos += 1;
        }
        fn skip_value(&mut self) {
            self.pos += 1;
        }
    }

    struct VecSink(Vec<([u8; 2], [u8; 1])>);
    impl FixedSink<2, 1> for VecSink {
        fn put(&mut self, key: &[u8; 2], value: &[u8; 1]) {
            self.0.push((*key, *value));
        }
    }

    fn source(keys: &[u16], tag: u8) -> VecSource {
        VecSource {
            records: keys.iter().map(|k| (k.to_be_bytes(), [tag])).collect(),
            pos: 0,
        }
    }

    #[test]
    fn merges_in_key_order_with_any_fanin() {
        for fanin in [0usize, 1, 2, 3, 16] {
            let inputs: Vec<Vec<u16>> = (0..fanin)
                .map(|i| {
                    (0..50)
                        .map(|j| (j * 7 + i * 3) as u16 % 300)
                        .collect::<Vec<_>>()
                })
                .map(|mut v| {
                    v.sort();
                    v
                })
                .collect();
            let sources: Vec<_> = inputs
                .iter()
                .enumerate()
                .map(|(i, keys)| source(keys, i as u8))
                .collect();
            let mut sink = VecSink(Vec::new());
            let n = merge_fixed(sources, &mut sink);

            let mut expected: Vec<u16> = inputs.concat();
            expected.sort();
            assert_eq!(n, expected.len());
            let got: Vec<u16> = sink.0.iter().map(|(k, _)| u16::from_be_bytes(*k)).collect();
            assert_eq!(got, expected, "fanin {fanin}");
        }
    }

    #[test]
    fn handles_empty_sources() {
        let sources = vec![source(&[], 0), source(&[1, 2], 1), source(&[], 2)];
        let mut sink = VecSink(Vec::new());
        assert_eq!(merge_fixed(sources, &mut sink), 2);
        assert_eq!(discard_fixed(vec![source(&[3], 0), source(&[1], 1)]), 2);
    }
}
