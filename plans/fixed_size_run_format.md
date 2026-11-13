# Plan: fixed-size run format with packed parallel output

## Goal

Sort files of fixed-size records (e.g. 8-byte key + 8-byte value, no delimiters)
and write the sorted result as a file in the same packed layout.

```
gen_sort_cli -i data.bin --key-size 8 --value-size 8 --run-format fixed -o sorted.bin \
  --run-gen-threads 8 --merge-threads 8 --rg-buf-mb 256 --merge-fanin 64 -v
```

- Run files and merge files are packed records: `[key K][value V]`, with no headers.
- The final merge writes each thread's range directly into `sorted.bin` at its exact
  offset (`pwrite`), so no extra pass is needed after the sort.
- The plain and OVC formats and existing commands behave as before.

## Design decisions

- **Sizes fixed at compile time:** `FixedRunFormat<const K: usize, const V: usize>`.
  `RunFormat` methods take no `self`, so const generics fit the trait without
  changing it. Supported sizes are listed in one dispatch macro: 8/8 and 10/90.
- **Run generation is unchanged:** replacement selection still uses the
  variable-size memory manager and writes through `append_from_kv`. Only the run
  files change.
- **Leave plain and OVC alone:** new trait items get default implementations.
- **Final output uses buffered I/O:** one `std::fs::File` without O_DIRECT, written
  with `write_at` (`pwrite`). Thread boundaries are multiples of `K+V`, not of
  4 KiB, so O_DIRECT would need block sharing between threads.
- **Durability:** the main thread calls `sync_all()` on the output after the merge
  threads finish, and that time counts toward merge time, so results stay
  comparable with the O_DIRECT writes of the other formats.

## Why each thread can compute its own offset

Partition boundaries are `(key, run_id, offset)` triples, which define a total
order over all records. Thread *i*'s lower bound is computed by the same code from
the same inputs as thread *i−1*'s upper bound, so ranges never overlap or leave
gaps. A thread's first output record is therefore at

```
start_i = Σ over input runs r of (records in r before lower_i)
```

In a fixed-size run, record *n* starts at byte `n × (K+V)`, so "records before a
bound" is found by seeking to the nearest sparse-index entry below it and
scanning forward at most one sparse interval. No coordination between threads is
needed.

## Milestone 1: replace the final-output flag with an enum

Pure refactor, no behaviour change.

- `engine.rs`: `pub enum FinalOutput { Runs, Discard, Packed(PathBuf) }`.
- Pass it in place of `discard_final_output: bool` through the `SorterCore` field,
  `multi_merge_with_hooks`, both final-merge call sites, `merge_once_with_hooks`,
  `SortHooks::merge_range` and its implementation in `run_format.rs`.
- Only the final merge receives the caller's choice; earlier merges get `Runs`.
- `set_discard_final_output(bool)` keeps its signature.

## Milestone 2: the fixed-size format

New module `src/sort/fixed/`, mirroring `src/sort/plain/`.

- `run.rs`, `FixedRun<K, V>`:
  - `append(key, value)` checks sizes (panics with a clear message on mismatch,
    matching how the other formats handle write errors) and samples into the
    sparse index with the same `should_sample` logic as plain.
  - Sparse index stores `(key, offset within run)` with the same meaning as plain,
    so all partitioning code works unchanged.
  - `position_of(bound)`: exact count of records before a `(key, run_id, offset)`
    bound.
  - `FixedRunIterator`: merge source (`next_key_into`, `copy_value_to`,
    `skip_value`) plus `Iterator<Item = (Vec<u8>, Vec<u8>)>`.
- `merge.rs`, `ZeroCopyMergeFixed<K, V>`: loser tree over `[u8; K]` keys; values are
  copied straight from the input reader to the output sink. Sinks: a `FixedRun`
  (intermediate merges) or the packed writer (final merge).
- `sorter.rs`, `FixedRunFormat<K, V>: RunFormat`, and
  `ExternalSorterFixed<K, V> = SorterCore<FormatSortHooks<FixedRunFormat<K, V>>>`.

## Milestone 3: packed output from the final merge

- `RunFormat` gains `const FIXED_RECORD_SIZE: Option<usize> = None` and
  `create_merge_sources_and_write_packed(...)` (default: error "not supported").
- Main thread, before the final merge: create the output file, `set_len(total ×
  (K+V))`.
- Each merge thread: `start = Σ position_of(lower)`, `end = Σ position_of(upper)`,
  merge into an 8 MB buffer flushed with `write_at`, then check it wrote exactly
  `(end − start) × (K+V)` bytes.
- After the threads join: `sync_all()`.
- Edge cases: a single run goes through the partitioned path (parallel copy);
  empty input creates an empty file.
- `PackedFileOutput` implements `SortOutput`; `iter()` reads the file back with
  `GenSortInputDirect::with_record_layout`, so `-v` keeps working.

## Milestone 4: benchmark runner and CLI

- `BenchmarkConfig.use_ovc` → `run_format: RunFormatChoice { Plain, Ovc,
  Fixed { key_size, value_size } }`, plus `output: Option<PathBuf>`.
- Runner dispatches with a `match`; `Fixed` goes through a macro over the
  compiled sizes.
- `gen_sort_cli`: `--run-format plain|ovc|fixed` (falls back to `--ovc` when not
  given) and `-o/--output`. Refuses: `-o` without `fixed`, `-o` with
  `--discard-final-output`, `-o` equal to `-i`, unsupported sizes.
- `kvbin_benchmark_cli` and `lineitem_benchmark_cli`: map `--ovc` to the enum.

## Tests

- `FixedRun`: round trip; `position_of` against brute force with bounds taken from
  other runs and many duplicate keys.
- End-to-end (`tests/test_fixed_format.rs`): random 8/8 data, output compared
  byte-for-byte with the input sorted in Rust. Covers 1/3/8 merge threads, all
  partition types, multi-pass merges, a single run, empty input, heavy
  duplicates, and a 10/90 case.
- CLI smoke test with `-o -v`.

## Risks

- `position_of` is the critical piece: an off-by-one makes threads overlap or leave
  gaps. Covered by brute-force tests and the exact-byte-count check.
- Run-generation memory per record is unchanged (future work: fixed-size run
  generation).
- Fixed is a new design point; report its benchmarks separately from plain/OVC.
- The planner still assumes plain/OVC overheads.
