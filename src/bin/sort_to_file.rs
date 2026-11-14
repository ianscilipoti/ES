//! Minimal entrypoint: sort a GenSort file with `ExternalSorterWithOVC` and
//! write the sorted records to an output file.
//!
//! Each output record is `key || value`, i.e. the same fixed-size layout as the
//! GenSort input, so the output is itself a valid GenSort file.
//!
//! Example:
//!   sort_to_file -i input.dat -o sorted.dat --dir /tmp \
//!     --run-gen-threads 8 --merge-threads 8 --rg-buf-mb 64 --merge-fanin 128 \
//!     --imbalance-factor 1.0 --partition-type size-balanced \
//!     --sparse-index-fraction 0.05

use clap::Parser;
use es::sort::core::engine::PartitionType;
use es::{ExternalSorterWithOVC, GenSortInputDirect, Sorter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const WRITE_BUFFER_SIZE: usize = 8 * 1024 * 1024;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PartitionArg {
    KeyOnly,
    CountBalanced,
    SizeBalanced,
}

impl From<PartitionArg> for PartitionType {
    fn from(value: PartitionArg) -> Self {
        match value {
            PartitionArg::KeyOnly => PartitionType::KeyOnly,
            PartitionArg::CountBalanced => PartitionType::CountBalanced,
            PartitionArg::SizeBalanced => PartitionType::SizeBalanced,
        }
    }
}

#[derive(Parser)]
#[command(about = "Sort a GenSort file with OVC external sort and write the result to a file")]
struct Args {
    /// Input GenSort file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path (sorted records, same layout as the input)
    #[arg(short, long)]
    output: PathBuf,

    /// Directory for temporary files
    #[arg(short, long)]
    dir: PathBuf,

    /// Threads for run generation
    #[arg(long)]
    run_gen_threads: usize,

    /// Threads for merge phase
    #[arg(long)]
    merge_threads: usize,

    /// RG buffer for run generation (MB)
    #[arg(long)]
    rg_buf_mb: f64,

    /// Merge fan-in (global per merge operation)
    #[arg(long)]
    merge_fanin: usize,

    /// Merge imbalance factor (>= 1.0)
    #[arg(long)]
    imbalance_factor: f64,

    /// Merge partition type (`key-only`, `count-balanced`, `size-balanced`)
    #[arg(long, value_name = "PARTITION")]
    partition_type: PartitionArg,

    /// Fraction of run-gen / merge memory reserved for sparse-index pages
    #[arg(long)]
    sparse_index_fraction: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input = GenSortInputDirect::new(&args.input)?;
    let num_records = input.len();

    // Unique temp directory under --dir; removed after the output is written
    // (the sorted output is read back from run files inside it).
    let temp_dir = args.dir.join(format!(
        "sort_to_file_{}_{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir)?;

    let result = sort_and_write(&args, input, &temp_dir);

    if let Err(e) = std::fs::remove_dir_all(&temp_dir) {
        eprintln!("Warning: failed to remove temp dir {temp_dir:?}: {e}");
    }

    let written = result?;
    if written != num_records {
        return Err(format!("record count mismatch: input {num_records}, output {written}").into());
    }
    Ok(())
}

fn sort_and_write(
    args: &Args,
    input: GenSortInputDirect,
    temp_dir: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let sort_start = Instant::now();
    let mut sorter = ExternalSorterWithOVC::new(
        args.run_gen_threads,
        (args.rg_buf_mb * 1024.0 * 1024.0) as usize,
        args.merge_threads,
        args.merge_fanin,
        temp_dir,
    );
    sorter.set_imbalance_factor(args.imbalance_factor);
    sorter.set_partition_type(args.partition_type.into());
    sorter.set_sparse_index_fraction(args.sparse_index_fraction);
    sorter.set_discard_final_output(false);

    let output = sorter.sort(Box::new(input))?;
    let sort_secs = sort_start.elapsed().as_secs_f64();

    let write_start = Instant::now();
    let file = File::create(&args.output)?;
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);
    let mut count: usize = 0;

    for (key, value) in output.iter() {
        writer.write_all(&key)?;
        writer.write_all(&value)?;
        count += 1;
    }

    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(output);
    let write_secs = write_start.elapsed().as_secs_f64();

    eprintln!(
        "Sorted {count} records -> {:?} (sort {sort_secs:.2}s, write {write_secs:.2}s)",
        args.output
    );
    Ok(count)
}
