//! Run files of fixed-size records.
//!
//! Layout: records back to back, each `[key: K bytes][value: V bytes]`, with no
//! headers or delimiters. Record `n` of a run starts at byte `n * (K + V)` from
//! the run's start, which is what lets a merge thread compute its exact
//! position in the final output.

use crate::diskio::aligned_reader::AlignedReader;
use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::file::SharedFd;
use crate::diskio::io_stats::IoStatsTracker;
use crate::sort::core::engine::RunSummary;
use crate::sort::core::run_format::{
    IndexingInterval, KeyRunIdOffsetBound, RUN_ID_COUNTER, cmp_key_run_offset,
};
use crate::sort::core::sparse_index::{SparseIndex, SparseIndexPagePool};
use crate::sort::fixed::merge::{FixedMergeSource, FixedSink};
use std::cell::UnsafeCell;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct FixedRun<const K: usize, const V: usize> {
    run_id: u32,
    fd: Arc<SharedFd>,
    writer: Option<AlignedWriter>,
    total_entries: usize,
    /// Absolute file offset where this run begins (run-generation runs share a
    /// file per thread).
    start_bytes: usize,
    sparse_index: UnsafeCell<SparseIndex>,
    sparse_index_refcount: AtomicUsize,
}

// SAFETY: same reasoning as the plain `Run`: the sparse index is mutated only
// during the single-threaded write phase, read concurrently during a merge, and
// cleared by the last reader after all readers have released it.
unsafe impl<const K: usize, const V: usize> Sync for FixedRun<K, V> {}

impl<const K: usize, const V: usize> FixedRun<K, V> {
    pub const RECORD_SIZE: usize = K + V;

    pub fn from_writer_with_indexing_interval(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self, String> {
        let run_id = RUN_ID_COUNTER.fetch_add(1, Ordering::AcqRel);
        Self::from_writer_with_indexing_interval_and_id(writer, indexing_interval, run_id)
    }

    pub fn from_writer_with_indexing_interval_and_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self, String> {
        const { assert!(K > 0, "fixed-size records need a non-empty key") };
        let start_bytes = writer.position() as usize;
        let fd = writer.get_fd();
        Ok(Self {
            run_id,
            fd,
            writer: Some(writer),
            total_entries: 0,
            start_bytes,
            sparse_index: UnsafeCell::new(SparseIndex::new(indexing_interval)),
            sparse_index_refcount: AtomicUsize::new(0),
        })
    }

    pub fn finalize_write(&mut self) -> AlignedWriter {
        self.writer.take().unwrap()
    }

    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    pub fn total_bytes(&self) -> usize {
        self.total_entries * Self::RECORD_SIZE
    }

    pub fn run_id(&self) -> u32 {
        self.run_id
    }

    pub fn start_key(&self) -> Option<&[u8]> {
        self.sparse_index().first_key()
    }

    pub fn sparse_index(&self) -> &SparseIndex {
        // SAFETY: no concurrent mutation during reads.
        unsafe { &*self.sparse_index.get() }
    }

    /// SAFETY: caller must ensure no concurrent access.
    #[allow(clippy::mut_from_ref)]
    pub fn sparse_index_mut(&self) -> &mut SparseIndex {
        unsafe { &mut *self.sparse_index.get() }
    }

    pub fn set_sparse_index_readers(&self, count: usize) {
        self.sparse_index_refcount.store(count, Ordering::Release);
    }

    pub fn release_sparse_index(&self) {
        if self.sparse_index_refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.sparse_index_mut().clear();
        }
    }

    pub fn release_sparse_index_to_pool(&self, pool: &SparseIndexPagePool) {
        if self.sparse_index_refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            pool.return_pages(self.sparse_index_mut().take_buffer());
        }
    }

    pub fn set_sparse_index_bootstrap(
        &mut self,
        avg_key_bytes: f64,
        avg_record_bytes: f64,
        sample_count: usize,
    ) {
        self.sparse_index
            .get_mut()
            .set_bootstrap(avg_key_bytes, avg_record_bytes, sample_count);
    }

    pub fn sparse_index_bootstrap(&self) -> Option<(f64, f64, usize)> {
        self.sparse_index().bootstrap()
    }

    /// Append one record given as slices. Panics if the sizes don't match the
    /// format, since a wrong-sized record would corrupt every later offset.
    pub fn append(&mut self, key: &[u8], value: &[u8]) {
        let key: &[u8; K] = key.try_into().unwrap_or_else(|_| {
            panic!(
                "fixed-size run format expects {}-byte keys, got {} bytes",
                K,
                key.len()
            )
        });
        let value: &[u8; V] = value.try_into().unwrap_or_else(|_| {
            panic!(
                "fixed-size run format expects {}-byte values, got {} bytes",
                V,
                value.len()
            )
        });
        self.put(key, value);
    }

    /// Relative offset of the last sparse-index entry that sorts before
    /// `bound`, or of entry 0 if none does. `None` if the run is empty.
    fn find_start_position(&self, bound: Option<(&[u8], u32, usize)>) -> Option<usize> {
        let index = self.sparse_index();
        if index.is_empty() {
            return None;
        }
        let Some(bound) = bound else {
            return Some(index.file_offset(0));
        };
        let mut best_idx = 0;
        let (mut left, mut right) = (0, index.len());
        while left < right {
            let mid = left + (right - left) / 2;
            let entry = (index.key(mid), self.run_id, index.file_offset(mid));
            if cmp_key_run_offset(entry, bound) == std::cmp::Ordering::Less {
                best_idx = mid;
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        Some(index.file_offset(best_idx))
    }

    /// Open a reader positioned at record `start_offset / RECORD_SIZE`.
    fn open_at(
        &self,
        start_offset: usize,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> FixedRunIterator<K, V> {
        debug_assert_eq!(start_offset % Self::RECORD_SIZE, 0);
        let mut reader = match io_tracker {
            Some(tracker) => AlignedReader::from_fd_with_tracer(self.fd.clone(), Some(tracker)),
            None => AlignedReader::from_fd(self.fd.clone()),
        }
        .expect("Failed to create run reader");
        let absolute = self.start_bytes + start_offset;
        if absolute > 0 {
            reader
                .seek(absolute as u64)
                .expect("Failed to seek in run file");
        }
        FixedRunIterator {
            reader,
            run_id: self.run_id,
            lower_inc: lower_inc.map(KeyRunIdOffsetBound::from_components),
            upper_exc: upper_exc.map(KeyRunIdOffsetBound::from_components),
            next_index: start_offset / Self::RECORD_SIZE,
            total_entries: self.total_entries,
            done: false,
        }
    }

    /// Merge source over the records in `[lower_inc, upper_exc)`.
    /// `None` for an empty run.
    pub fn scan_range_as_source(
        &self,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Option<FixedRunIterator<K, V>> {
        let start = self.find_start_position(lower_inc)?;
        Some(self.open_at(start, lower_inc, upper_exc, io_tracker))
    }

    /// Exact number of records in this run that sort before `bound` in
    /// `(key, run_id, offset)` order: the same order the merge uses to split
    /// work between threads. Reads at most one sparse-index interval.
    pub fn entries_before(
        &self,
        bound: (&[u8], u32, usize),
        io_tracker: Option<IoStatsTracker>,
    ) -> usize {
        if self.total_entries == 0 {
            return 0;
        }
        // Fast path: the first record (always sampled, at offset 0) is not
        // before the bound, so no record is.
        if let Some(first) = self.start_key() {
            if cmp_key_run_offset((first, self.run_id, 0), bound) != std::cmp::Ordering::Less {
                return 0;
            }
        }
        let start = self
            .find_start_position(Some(bound))
            .expect("non-empty run has a sparse index");
        let mut iter = self.open_at(start, None, Some(bound), io_tracker);
        let mut key = [0u8; K];
        let mut count = start / Self::RECORD_SIZE;
        while iter.next_key(&mut key) {
            iter.skip_value();
            count += 1;
        }
        count
    }
}

impl<const K: usize, const V: usize> FixedSink<K, V> for FixedRun<K, V> {
    #[inline]
    fn put(&mut self, key: &[u8; K], value: &[u8; V]) {
        let sparse_index = self.sparse_index.get_mut();
        let should_sample =
            sparse_index.should_sample(self.total_entries, self.total_entries * Self::RECORD_SIZE);
        let record_index = self.total_entries;
        let record_offset = record_index * Self::RECORD_SIZE;

        let writer = self.writer.as_mut().expect("FixedRun has no writer");
        writer.write_all(key).expect("Failed to write key");
        writer.write_all(value).expect("Failed to write value");
        self.total_entries += 1;

        let sparse_index = self.sparse_index.get_mut();
        sparse_index.observe_record(K, Self::RECORD_SIZE);
        if should_sample {
            sparse_index.record_sample(key, record_offset, record_index, record_offset);
        }
    }
}

impl<const K: usize, const V: usize> RunSummary for FixedRun<K, V> {
    fn total_entries(&self) -> usize {
        self.total_entries
    }

    fn total_bytes(&self) -> usize {
        self.total_entries * Self::RECORD_SIZE
    }
}

/// Reads the records of one fixed-size run in order, optionally limited to a
/// `[lower_inc, upper_exc)` range of `(key, run_id, offset)` bounds.
pub struct FixedRunIterator<const K: usize, const V: usize> {
    reader: AlignedReader,
    run_id: u32,
    lower_inc: Option<KeyRunIdOffsetBound>,
    upper_exc: Option<KeyRunIdOffsetBound>,
    /// Index (within the run) of the next record to read.
    next_index: usize,
    total_entries: usize,
    done: bool,
}

impl<const K: usize, const V: usize> FixedMergeSource<K, V> for FixedRunIterator<K, V> {
    /// Read the next in-range key. On `true`, the reader is positioned at that
    /// record's value, which must be consumed with `read_value` or
    /// `skip_value` before the next call.
    #[inline]
    fn next_key(&mut self, key: &mut [u8; K]) -> bool {
        loop {
            if self.done || self.next_index >= self.total_entries {
                self.done = true;
                return false;
            }
            let offset = self.next_index * (K + V);
            self.reader
                .read_exact(key)
                .expect("Failed to read key from run");
            self.next_index += 1;

            if let Some(lb) = &self.lower_inc {
                if lb.lt(key, self.run_id, offset) {
                    self.skip_value();
                    continue;
                }
                // Records in a run are in bound order, so once one is at or
                // past the lower bound every later one is too.
                self.lower_inc = None;
            }
            if let Some(ub) = &self.upper_exc {
                if ub.ge(key, self.run_id, offset) {
                    self.done = true;
                    return false;
                }
            }
            return true;
        }
    }

    #[inline]
    fn read_value(&mut self, value: &mut [u8; V]) {
        self.reader
            .read_exact(value)
            .expect("Failed to read value from run");
    }

    #[inline]
    fn skip_value(&mut self) {
        let mut scratch = [0u8; V];
        self.read_value(&mut scratch);
    }
}

impl<const K: usize, const V: usize> Iterator for FixedRunIterator<K, V> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut key = [0u8; K];
        if !self.next_key(&mut key) {
            return None;
        }
        let mut value = [0u8; V];
        self.read_value(&mut value);
        Some((key.to_vec(), value.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand::small_thread_rng;
    use rand::Rng;
    use tempfile::TempDir;

    type Run8 = FixedRun<8, 8>;

    fn new_run(dir: &TempDir, name: &str, stride: usize) -> Run8 {
        let fd = Arc::new(SharedFd::new_from_path(dir.path().join(name), true).unwrap());
        let writer = AlignedWriter::from_fd(fd).unwrap();
        Run8::from_writer_with_indexing_interval(writer, IndexingInterval::records(stride)).unwrap()
    }

    fn finish(run: &mut Run8) {
        drop(run.finalize_write());
    }

    fn sorted_keys(n: usize, distinct: u64) -> Vec<u64> {
        let mut rng = small_thread_rng();
        let mut keys: Vec<u64> = (0..n).map(|_| rng.random_range(0..distinct)).collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn round_trip_and_range_scan() {
        let dir = TempDir::new().unwrap();
        let mut run = new_run(&dir, "rt.dat", 7);
        for i in 0..1000u64 {
            run.append(&i.to_be_bytes(), &(!i).to_be_bytes());
        }
        finish(&mut run);
        assert_eq!(run.total_entries(), 1000);
        assert_eq!(run.total_bytes(), 16_000);

        let all: Vec<_> = run
            .scan_range_as_source(None, None, None)
            .unwrap()
            .collect();
        assert_eq!(all.len(), 1000);
        for (i, (k, v)) in all.iter().enumerate() {
            assert_eq!(k.as_slice(), &(i as u64).to_be_bytes());
            assert_eq!(v.as_slice(), &(!(i as u64)).to_be_bytes());
        }

        let lo = 250u64.to_be_bytes();
        let hi = 700u64.to_be_bytes();
        let ranged: Vec<_> = run
            .scan_range_as_source(Some((&lo, 0, 0)), Some((&hi, 0, 0)), None)
            .unwrap()
            .collect();
        assert_eq!(ranged.len(), 450);
        assert_eq!(ranged[0].0.as_slice(), &lo);
    }

    #[test]
    #[should_panic(expected = "expects 8-byte keys")]
    fn wrong_key_size_panics() {
        let dir = TempDir::new().unwrap();
        let mut run = new_run(&dir, "bad.dat", 10);
        run.append(&[1, 2, 3], &[0u8; 8]);
    }

    /// `entries_before` must agree with a brute-force count for bounds taken
    /// from other runs (different run_id), from this run itself, and for keys
    /// with many duplicates.
    #[test]
    fn entries_before_matches_brute_force() {
        let dir = TempDir::new().unwrap();
        let runs_keys: Vec<Vec<u64>> = (0..3).map(|_| sorted_keys(3000, 200)).collect();
        let mut runs = Vec::new();
        for (r, keys) in runs_keys.iter().enumerate() {
            let mut run = new_run(&dir, &format!("r{r}.dat"), 37);
            for k in keys {
                run.append(&k.to_be_bytes(), &[0u8; 8]);
            }
            finish(&mut run);
            runs.push(run);
        }

        let brute = |run: &Run8, keys: &[u64], bound: (&[u8], u32, usize)| {
            keys.iter()
                .enumerate()
                .filter(|(i, k)| {
                    cmp_key_run_offset((&k.to_be_bytes(), run.run_id(), i * 16), bound)
                        == std::cmp::Ordering::Less
                })
                .count()
        };

        for source in &runs {
            let index = source.sparse_index();
            for e in 0..index.len() {
                let bound = (index.key(e), source.run_id(), index.file_offset(e));
                for (run, keys) in runs.iter().zip(&runs_keys) {
                    assert_eq!(
                        run.entries_before(bound, None),
                        brute(run, keys, bound),
                        "bound from run {} entry {}",
                        source.run_id(),
                        e
                    );
                }
            }
        }
        // Key-only bounds (run_id 0, offset 0) and bounds past every key.
        for key in [0u64, 1, 57, 199, 200, u64::MAX] {
            let kb = key.to_be_bytes();
            for (run, keys) in runs.iter().zip(&runs_keys) {
                assert_eq!(
                    run.entries_before((&kb, 0, 0), None),
                    brute(run, keys, (&kb, 0, 0))
                );
            }
        }
    }
}
