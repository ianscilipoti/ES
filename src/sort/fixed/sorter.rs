//! `RunFormat` implementation for fixed-size records.

use std::sync::Barrier;

use crate::diskio::aligned_writer::AlignedWriter;
use crate::diskio::io_stats::IoStatsTracker;
use crate::sort::core::engine::SorterCore;
use crate::sort::core::run_format::{
    FormatSortHooks, IndexingInterval, MergeableRun, RunFormat, RunsOutput, cmp_key_run_offset,
};
use crate::sort::core::sparse_index::{SparseIndex, SparseIndexPage, SparseIndexPagePool};
use crate::sort::fixed::merge::{PackedWriter, discard_fixed, merge_fixed};
use crate::sort::fixed::run::{FixedRun, FixedRunIterator};
use crate::sort::plain::merge::MergeIterator;

/// Run format whose records are all `K`-byte keys followed by `V`-byte values.
#[derive(Clone, Copy, Default)]
pub struct FixedRunFormat<const K: usize, const V: usize>;

pub type FixedSortHooks<const K: usize, const V: usize> = FormatSortHooks<FixedRunFormat<K, V>>;
pub type ExternalSorterFixed<const K: usize, const V: usize> = SorterCore<FixedSortHooks<K, V>>;
pub type FixedMergeableRun<const K: usize, const V: usize> = MergeableRun<FixedRunFormat<K, V>>;
pub type FixedRunsOutput<const K: usize, const V: usize> = RunsOutput<FixedRunFormat<K, V>>;

/// Record layouts `(key_size, value_size)` compiled into the fixed format.
/// To support another size, add it here and in `dispatch_fixed_layout!`.
pub const SUPPORTED_FIXED_LAYOUTS: &[(usize, usize)] = &[(8, 8), (10, 90)];

/// Call a generic function `f::<K, V>(args...)` for a runtime `(key_size,
/// value_size)` pair, choosing among the layouts in `SUPPORTED_FIXED_LAYOUTS`.
/// Evaluates to `Result<return type of f, String>`.
#[macro_export]
macro_rules! dispatch_fixed_layout {
    ($key_size:expr, $value_size:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match ($key_size, $value_size) {
            (8, 8) => Ok($f::<8, 8>($($arg),*)),
            (10, 90) => Ok($f::<10, 90>($($arg),*)),
            (k, v) => Err(format!(
                "fixed run format is not compiled for {}-byte keys / {}-byte values; supported: {}",
                k,
                v,
                $crate::sort::fixed::sorter::SUPPORTED_FIXED_LAYOUTS
                    .iter()
                    .map(|(k, v)| format!("{}/{}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    };
}

const DEFAULT_RUN_INDEXING_INTERVAL: usize = 100;

type Bound<'a> = Option<(&'a [u8], u32, usize)>;

/// Merge sources for the records of `runs` that fall in `[lower, upper)`.
/// For range-partitioned runs, only the sub-runs that can overlap the range
/// are opened (same selection as the plain format).
fn build_sources<const K: usize, const V: usize>(
    runs: &[FixedMergeableRun<K, V>],
    lower: Bound<'_>,
    upper: Bound<'_>,
    io_tracker: &Option<IoStatsTracker>,
) -> Vec<FixedRunIterator<K, V>> {
    let mut sources = Vec::new();
    for run in runs {
        match run {
            MergeableRun::Single(r) => {
                sources.extend(r.scan_range_as_source(lower, upper, io_tracker.clone()));
            }
            MergeableRun::RangePartitioned(sub_runs) => {
                let non_empty: Vec<&FixedRun<K, V>> = sub_runs
                    .iter()
                    .filter(|r| r.start_key().is_some())
                    .collect();
                if non_empty.is_empty() {
                    continue;
                }
                fn start_key<const K: usize, const V: usize>(
                    r: &FixedRun<K, V>,
                ) -> (&[u8], u32, usize) {
                    (r.start_key().unwrap(), r.run_id(), 0)
                }
                // Last sub-run that starts before `lower` (or 0).
                let first = match lower {
                    None => 0,
                    Some(lb) => non_empty
                        .partition_point(|r| {
                            cmp_key_run_offset(start_key(r), lb) == std::cmp::Ordering::Less
                        })
                        .saturating_sub(1),
                };
                // Sub-runs that start before `upper`.
                let end = match upper {
                    None => non_empty.len(),
                    Some(ub) => non_empty.partition_point(|r| {
                        cmp_key_run_offset(start_key(r), ub) == std::cmp::Ordering::Less
                    }),
                };
                for sub_run in non_empty.iter().take(end).skip(first) {
                    sources.extend(sub_run.scan_range_as_source(lower, upper, io_tracker.clone()));
                }
            }
            MergeableRun::StatsOnly { .. } => {}
        }
    }
    sources
}

/// Number of records across `runs` that sort before `bound`. `None` as a lower
/// bound means "start of everything" (0); as an upper bound, "end of
/// everything" (all records).
fn entries_before_all<const K: usize, const V: usize>(
    runs: &[FixedMergeableRun<K, V>],
    bound: Bound<'_>,
    none_means_end: bool,
    io_tracker: &Option<IoStatsTracker>,
) -> usize {
    let Some(bound) = bound else {
        return if none_means_end {
            runs.iter().map(|r| r.total_entries()).sum()
        } else {
            0
        };
    };
    runs.iter()
        .map(|run| match run {
            MergeableRun::Single(r) => r.entries_before(bound, io_tracker.clone()),
            MergeableRun::RangePartitioned(sub_runs) => sub_runs
                .iter()
                .map(|r| r.entries_before(bound, io_tracker.clone()))
                .sum(),
            MergeableRun::StatsOnly { .. } => 0,
        })
        .sum()
}

impl<const K: usize, const V: usize> RunFormat for FixedRunFormat<K, V> {
    type Run = FixedRun<K, V>;
    type Record = (Vec<u8>, Vec<u8>);
    type AppendState = ();

    const FIXED_RECORD_LAYOUT: Option<(usize, usize)> = Some((K, V));

    fn new_run(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
    ) -> Result<Self::Run, String> {
        FixedRun::from_writer_with_indexing_interval(writer, indexing_interval)
    }

    fn create_merge_run_with_id(
        writer: AlignedWriter,
        indexing_interval: IndexingInterval,
        run_id: u32,
    ) -> Result<Self::Run, String> {
        FixedRun::from_writer_with_indexing_interval_and_id(writer, indexing_interval, run_id)
    }

    fn finalize_run(run: &mut Self::Run) -> AlignedWriter {
        run.finalize_write()
    }

    fn append_from_kv(
        _state: &mut Self::AppendState,
        run: &mut Self::Run,
        key: &[u8],
        value: &[u8],
    ) {
        run.append(key, value);
    }

    fn scan_range(
        run: &Self::Run,
        lower_inc: Option<(&[u8], u32, usize)>,
        upper_exc: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        match run.scan_range_as_source(lower_inc, upper_exc, io_tracker) {
            Some(iter) => Box::new(iter),
            None => Box::new(std::iter::empty()),
        }
    }

    fn merge_iterators(
        iterators: Vec<Box<dyn Iterator<Item = Self::Record> + Send>>,
    ) -> Box<dyn Iterator<Item = Self::Record> + Send> {
        Box::new(MergeIterator::new(iterators))
    }

    fn run_id(run: &Self::Run) -> u32 {
        run.run_id()
    }

    fn sparse_index<'a>(run: &'a Self::Run) -> &'a SparseIndex {
        run.sparse_index()
    }

    fn start_key<'a>(run: &'a Self::Run) -> Option<(&'a [u8], u32, usize)> {
        run.start_key().map(|key| (key, run.run_id(), 0))
    }

    fn record_key<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.0
    }

    fn record_value<'a>(record: &'a Self::Record) -> &'a [u8] {
        &record.1
    }

    fn record_into_kv(record: Self::Record) -> (Vec<u8>, Vec<u8>) {
        record
    }

    fn set_sparse_index_bootstrap(
        run: &mut Self::Run,
        avg_key_bytes: f64,
        avg_record_bytes: f64,
        sample_count: usize,
    ) {
        run.set_sparse_index_bootstrap(avg_key_bytes, avg_record_bytes, sample_count);
    }

    fn sparse_index_bootstrap(run: &Self::Run) -> Option<(f64, f64, usize)> {
        run.sparse_index_bootstrap()
    }

    fn set_sparse_index_readers(run: &Self::Run, count: usize) {
        run.set_sparse_index_readers(count);
    }

    fn release_sparse_index(run: &Self::Run) {
        run.release_sparse_index();
    }

    fn release_sparse_index_to_pool(run: &Self::Run, pool: &SparseIndexPagePool) {
        run.release_sparse_index_to_pool(pool);
    }

    fn seed_sparse_index_buffer(run: &mut Self::Run, pages: Vec<SparseIndexPage>) {
        run.sparse_index_mut().seed_buffer(pages);
    }

    fn sparse_index_budget_pages(run: &Self::Run) -> usize {
        run.sparse_index().budget_pages()
    }

    fn create_merge_sources_and_execute(
        runs: &[MergeableRun<Self>],
        output_run: &mut Self::Run,
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
        page_pool: &SparseIndexPagePool,
        barrier: &Barrier,
        thread_id: usize,
        merge_threads: usize,
    ) {
        let sources = build_sources(runs, lower_bound, upper_bound, &io_tracker);

        for run in runs {
            run.release_sparse_index_to_pool(page_pool);
        }
        barrier.wait();

        // Hand this thread its share of the freed sparse-index pages for the
        // output run's own index (same policy as the plain format).
        let total = page_pool.len();
        let budget_cap = output_run.sparse_index().budget_pages();
        let fair_share = total / merge_threads + usize::from(thread_id < total % merge_threads);
        let pages = page_pool.take(fair_share.min(budget_cap));
        output_run.sparse_index_mut().seed_buffer(pages);

        merge_fixed(sources, output_run);
    }

    fn create_merge_sources_and_discard(
        runs: &[MergeableRun<Self>],
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
        page_pool: &SparseIndexPagePool,
        barrier: &Barrier,
        _thread_id: usize,
    ) -> usize {
        let sources = build_sources(runs, lower_bound, upper_bound, &io_tracker);
        for run in runs {
            run.release_sparse_index_to_pool(page_pool);
        }
        barrier.wait();
        discard_fixed(sources)
    }

    fn create_merge_sources_and_write_packed(
        runs: &[MergeableRun<Self>],
        lower_bound: Option<(&[u8], u32, usize)>,
        upper_bound: Option<(&[u8], u32, usize)>,
        io_tracker: Option<IoStatsTracker>,
        page_pool: &SparseIndexPagePool,
        barrier: &Barrier,
        thread_id: usize,
        output: &std::fs::File,
    ) -> Result<usize, String> {
        // Everything that needs the sparse indexes happens before they are
        // released: opening the sources and locating this range's exact
        // position in the sorted order.
        let sources = build_sources(runs, lower_bound, upper_bound, &io_tracker);
        let start = entries_before_all(runs, lower_bound, false, &io_tracker);
        let end = entries_before_all(runs, upper_bound, true, &io_tracker);

        for run in runs {
            run.release_sparse_index_to_pool(page_pool);
        }
        barrier.wait();

        let record_size = (K + V) as u64;
        let mut writer = PackedWriter::new(output, start as u64 * record_size, io_tracker);
        let entries = merge_fixed(sources, &mut writer);
        let written = writer.finish()?;

        let expected = end.saturating_sub(start);
        if entries != expected || written != expected as u64 * record_size {
            return Err(format!(
                "Merge thread {} wrote {} records ({} bytes) but its range [{}, {}) holds {} records; \
                 partition boundaries and record counts disagree",
                thread_id, entries, written, start, end, expected
            ));
        }
        Ok(entries)
    }

    fn run_replacement_selection<
        S: crate::sort::run_sink::RunSink<MergeableRun = MergeableRun<Self>>,
    >(
        scanner: crate::sort::core::engine::Scanner,
        sink: &mut S,
        data_bytes: usize,
    ) -> crate::replacement_selection::ReplacementSelectionStats {
        crate::replacement_selection::run_replacement_selection_mm(scanner, sink, data_bytes)
    }
}

impl<const K: usize, const V: usize> SorterCore<FixedSortHooks<K, V>> {
    pub fn new(
        run_gen_threads: usize,
        run_gen_mem: usize,
        merge_threads: usize,
        merge_fanin: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        SorterCore::with_indexing_interval(
            run_gen_threads,
            run_gen_mem,
            merge_threads,
            merge_fanin,
            DEFAULT_RUN_INDEXING_INTERVAL,
            base_dir,
        )
    }
}
