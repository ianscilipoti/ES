//! End-to-end tests for the fixed-size run format and packed output.

use es::sort::core::engine::PartitionType;
use es::sort::fixed::sorter::ExternalSorterFixed;
use es::{ExternalSorter, GenSortInputDirect, InMemInput, Sorter};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

type Record = (Vec<u8>, Vec<u8>);

fn random_records<const K: usize, const V: usize>(
    n: usize,
    distinct: u64,
    seed: u64,
) -> Vec<Record> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let k = rng.random_range(0..distinct.max(1));
            let mut key = vec![0u8; K];
            let kb = k.to_be_bytes();
            // Put the random part at the end of the key so a K-byte key sorts by it.
            let n = kb.len().min(K);
            key[K - n..].copy_from_slice(&kb[kb.len() - n..]);
            let value: Vec<u8> = (0..V).map(|_| rng.random()).collect();
            (key, value)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Config {
    run_gen_threads: usize,
    run_gen_mem: usize,
    merge_threads: usize,
    fanin: usize,
    partition: PartitionType,
}

const SMALL: Config = Config {
    run_gen_threads: 2,
    run_gen_mem: 128 * 1024, // below 64 KiB replacement selection is unbounded
    merge_threads: 3,
    fanin: 64,
    partition: PartitionType::SizeBalanced,
};

/// Sort `data` with packed output and check the file holds exactly the input
/// records, in key order.
fn check_packed<const K: usize, const V: usize>(data: Vec<Record>, cfg: Config) {
    let dir = TempDir::new().unwrap();
    let out_path = dir.path().join("sorted.bin");

    let mut sorter = ExternalSorterFixed::<K, V>::new(
        cfg.run_gen_threads,
        cfg.run_gen_mem,
        cfg.merge_threads,
        cfg.fanin,
        dir.path(),
    );
    sorter.set_partition_type(cfg.partition);
    sorter.set_packed_output(&out_path);
    let output = sorter
        .sort(Box::new(InMemInput { data: data.clone() }))
        .unwrap();

    let bytes = std::fs::read(&out_path).unwrap();
    assert_eq!(bytes.len(), data.len() * (K + V), "output size, {cfg:?}");
    let got: Vec<Record> = bytes
        .chunks_exact(K + V)
        .map(|r| (r[..K].to_vec(), r[K..].to_vec()))
        .collect();

    for w in got.windows(2) {
        assert!(w[0].0 <= w[1].0, "keys out of order, {cfg:?}");
    }
    // Same records as the input (equal keys may come out in any order).
    let mut expected = data;
    expected.sort();
    let mut got_sorted = got.clone();
    got_sorted.sort();
    assert_eq!(got_sorted, expected, "records differ from input, {cfg:?}");

    // The returned SortOutput reads the same file back.
    let via_output: Vec<Record> = output.iter().collect();
    assert_eq!(via_output, got);

    // And it is a valid fixed-record file for the input reader.
    let reread = GenSortInputDirect::with_record_layout(&out_path, K, V).unwrap();
    assert_eq!(reread.len(), got.len());
}

#[test]
fn packed_8_8_across_threads_and_partition_types() {
    let data = random_records::<8, 8>(40_000, u64::MAX, 1);
    for merge_threads in [1, 3, 8] {
        for partition in [
            PartitionType::KeyOnly,
            PartitionType::CountBalanced,
            PartitionType::SizeBalanced,
        ] {
            check_packed::<8, 8>(
                data.clone(),
                Config {
                    merge_threads,
                    partition,
                    ..SMALL
                },
            );
        }
    }
}

#[test]
fn packed_8_8_multi_pass_merge() {
    // Small fan-in forces intermediate merges, whose range-partitioned outputs
    // become inputs of the final packed merge.
    let data = random_records::<8, 8>(60_000, u64::MAX, 2);
    for fanin in [2, 3, 5] {
        check_packed::<8, 8>(
            data.clone(),
            Config {
                fanin,
                merge_threads: 4,
                ..SMALL
            },
        );
    }
}

#[test]
fn packed_8_8_heavy_duplicates() {
    for distinct in [1, 3, 50] {
        let data = random_records::<8, 8>(30_000, distinct, 3);
        for partition in [PartitionType::KeyOnly, PartitionType::SizeBalanced] {
            check_packed::<8, 8>(
                data.clone(),
                Config {
                    merge_threads: 5,
                    fanin: 4,
                    partition,
                    ..SMALL
                },
            );
        }
    }
}

#[test]
fn packed_8_8_single_run() {
    // One run-generation thread with plenty of memory: no merge is needed, but
    // the packed output must still be written.
    let data = random_records::<8, 8>(5_000, u64::MAX, 4);
    check_packed::<8, 8>(
        data,
        Config {
            run_gen_threads: 1,
            run_gen_mem: 16 << 20,
            merge_threads: 4,
            ..SMALL
        },
    );
}

#[test]
fn packed_8_8_empty_and_tiny_inputs() {
    for n in [0, 1, 2, 7] {
        check_packed::<8, 8>(random_records::<8, 8>(n, u64::MAX, 5), SMALL);
    }
}

#[test]
fn packed_10_90() {
    let data = random_records::<10, 90>(20_000, u64::MAX, 6);
    check_packed::<10, 90>(
        data,
        Config {
            merge_threads: 4,
            fanin: 4,
            ..SMALL
        },
    );
}

#[test]
fn fixed_format_without_packed_output_sorts_into_runs() {
    let dir = TempDir::new().unwrap();
    let data = random_records::<8, 8>(30_000, 1000, 7);
    let mut sorter = ExternalSorterFixed::<8, 8>::new(2, 128 * 1024, 3, 4, dir.path());
    let output = sorter
        .sort(Box::new(InMemInput { data: data.clone() }))
        .unwrap();
    let got: Vec<Record> = output.iter().collect();
    for w in got.windows(2) {
        assert!(w[0].0 <= w[1].0);
    }
    let mut expected = data;
    expected.sort();
    let mut got_sorted = got;
    got_sorted.sort();
    assert_eq!(got_sorted, expected);
}

#[test]
fn fixed_format_rejects_wrong_record_size() {
    let dir = TempDir::new().unwrap();
    let mut sorter = ExternalSorterFixed::<8, 8>::new(1, 64 * 1024, 1, 4, dir.path());
    let data = vec![(vec![1u8; 8], vec![2u8; 8]), (vec![1u8; 4], vec![2u8; 8])];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sorter.sort(Box::new(InMemInput { data }))
    }));
    assert!(
        !matches!(result, Ok(Ok(_))),
        "a 4-byte key must not be accepted by the 8/8 format"
    );
}

#[test]
fn packed_output_rejected_for_variable_size_formats() {
    let dir = TempDir::new().unwrap();
    let mut sorter = ExternalSorter::new(1, 64 * 1024, 1, 4, dir.path());
    sorter.set_packed_output(dir.path().join("out.bin"));
    let err = sorter
        .sort(Box::new(InMemInput {
            data: vec![(b"k".to_vec(), b"v".to_vec())],
        }))
        .err()
        .expect("plain format must reject packed output");
    assert!(err.contains("fixed-size"), "{err}");
}
