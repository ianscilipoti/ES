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
//!     --sparse-index-fraction 0.05 --write-output true
//!
//! Planner example (derives threads, rg buffer and fan-in from the budget):
//!   sort_to_file -i input.dat -o sorted.dat --dir /tmp \
//!     --use-planner --memory-mb 2048 --max-threads 16 \
//!     --sparse-index-fraction 0.05 --write-output true

use clap::Parser;
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::sort::core::engine::PartitionType;
use es::sort_policy_sub::{PlannerConfig, plan_resource_efficient};
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

    /// Derive run_gen_threads, merge_threads, rg_buf_mb and merge_fanin from
    /// the dataset size and the budget (--memory-mb, --max-threads). When
    /// --write-output is true, the output write buffer is subtracted from
    /// --memory-mb before planning.
    #[arg(long, default_value = "false")]
    use_planner: bool,

    /// Compute and print the planner's configuration, then exit without
    /// sorting. Requires --memory-mb and --max-threads.
    #[arg(long, default_value = "false")]
    print_plan: bool,

    /// Total memory budget in MB (used by --use-planner / --print-plan)
    #[arg(long, required_if_eq_any = [("use_planner", "true"), ("print_plan", "true")])]
    memory_mb: Option<f64>,

    /// Maximum thread count (used by --use-planner / --print-plan)
    #[arg(long, required_if_eq_any = [("use_planner", "true"), ("print_plan", "true")])]
    max_threads: Option<usize>,

    /// Threads for run generation
    #[arg(long, required_unless_present_any = ["use_planner", "print_plan"])]
    run_gen_threads: Option<usize>,

    /// Threads for merge phase
    #[arg(long, required_unless_present_any = ["use_planner", "print_plan"])]
    merge_threads: Option<usize>,

    /// RG buffer for run generation (MB)
    #[arg(long, required_unless_present_any = ["use_planner", "print_plan"])]
    rg_buf_mb: Option<f64>,

    /// Merge fan-in (global per merge operation)
    #[arg(long, required_unless_present_any = ["use_planner", "print_plan"])]
    merge_fanin: Option<usize>,

    /// Merge imbalance factor (>= 1.0)
    #[arg(long, default_value = "1.0")]
    imbalance_factor: f64,

    /// Merge partition type (`key-only`, `count-balanced`, `size-balanced`)
    #[arg(long, default_value = "size-balanced", value_name = "PARTITION")]
    partition_type: PartitionArg,

    /// Fraction of run-gen / merge memory reserved for sparse-index pages
    #[arg(long)]
    sparse_index_fraction: f64,

    /// Copy the sorted output into --output (`true`/`false`). Either way, the
    /// final merge runs normally: each merge thread writes its sorted
    /// key-range segment to its own file in the temp dir. When false, the
    /// serial copy of those segments into --output is skipped, so no output
    /// file is created (mimics threads writing segments to a shared file).
    #[arg(long, action = clap::ArgAction::Set)]
    write_output: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input = GenSortInputDirect::new(&args.input)?;
    let num_records = input.len();

    let Some(config) = resolve_config(&args, &input)? else {
        return Ok(()); // --print-plan
    };

    // Unique temp directory under --dir; removed after the output is written
    // (the sorted output is read back from run files inside it).
    let temp_dir = args.dir.join(format!(
        "sort_to_file_{}_{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir)?;

    let result = sort_and_write(&args, &config, input, &temp_dir);

    if let Err(e) = std::fs::remove_dir_all(&temp_dir) {
        eprintln!("Warning: failed to remove temp dir {temp_dir:?}: {e}");
    }

    if let Some(written) = result? {
        if written != num_records {
            return Err(
                format!("record count mismatch: input {num_records}, output {written}").into(),
            );
        }
    }
    Ok(())
}

/// Sort parameters, from either the explicit CLI args or the planner.
struct ResolvedConfig {
    run_gen_threads: usize,
    merge_threads: usize,
    rg_buf_mb: f64,
    merge_fanin: usize,
}

/// Returns `None` when `--print-plan` was given (plan printed, nothing to sort).
fn resolve_config(
    args: &Args,
    input: &GenSortInputDirect,
) -> Result<Option<ResolvedConfig>, Box<dyn std::error::Error>> {
    if !args.use_planner && !args.print_plan {
        // clap's required_unless_present_any guarantees these are set.
        return Ok(Some(ResolvedConfig {
            run_gen_threads: args.run_gen_threads.unwrap(),
            merge_threads: args.merge_threads.unwrap(),
            rg_buf_mb: args.rg_buf_mb.unwrap(),
            merge_fanin: args.merge_fanin.unwrap(),
        }));
    }

    let memory_mb = args.memory_mb.ok_or("--memory-mb is required with the planner")?;
    let max_threads = args.max_threads.ok_or("--max-threads is required with the planner")?;
    let dataset_mb = input.file_size()? as f64 / (1024.0 * 1024.0);

    // The planner only budgets run generation and merge; reserve the output
    // write buffer out of the total when we will copy to --output.
    let write_reserve_mb = if args.write_output {
        WRITE_BUFFER_SIZE as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    };
    let planner_memory_mb = memory_mb - write_reserve_mb;
    if planner_memory_mb <= 0.0 {
        return Err(format!(
            "--memory-mb {memory_mb} leaves no memory after reserving {write_reserve_mb} MB for the output write buffer"
        )
        .into());
    }

    let plan = plan_resource_efficient(&PlannerConfig {
        dataset_mb,
        memory_mb: planner_memory_mb,
        max_threads,
        page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
        sparse_index_fraction: args.sparse_index_fraction,
        ..PlannerConfig::default()
    });
    eprintln!(
        "Planner: dataset {dataset_mb:.2} MB, budget {memory_mb:.1} MB \
         ({planner_memory_mb:.1} MB after {write_reserve_mb:.1} MB write reserve)"
    );
    eprintln!("{plan}");

    if args.print_plan {
        return Ok(None);
    }
    Ok(Some(ResolvedConfig {
        run_gen_threads: plan.run_gen_threads,
        merge_threads: plan.merge_threads,
        rg_buf_mb: plan.rg_buf_mb,
        merge_fanin: plan.merge_fanin,
    }))
}

/// Returns the number of records written, or `None` if output was discarded.
fn sort_and_write(
    args: &Args,
    config: &ResolvedConfig,
    input: GenSortInputDirect,
    temp_dir: &Path,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let sort_start = Instant::now();
    let mut sorter = ExternalSorterWithOVC::new(
        config.run_gen_threads,
        (config.rg_buf_mb * 1024.0 * 1024.0) as usize,
        config.merge_threads,
        config.merge_fanin,
        temp_dir,
    );
    sorter.set_imbalance_factor(args.imbalance_factor);
    sorter.set_partition_type(args.partition_type.into());
    sorter.set_sparse_index_fraction(args.sparse_index_fraction);
    // Always write the final merge: each merge thread writes its segment.
    sorter.set_discard_final_output(false);

    let output = sorter.sort(Box::new(input))?;
    let sort_secs = sort_start.elapsed().as_secs_f64();

    if !args.write_output {
        drop(output);
        eprintln!("Sorted; per-thread segments written to temp dir, not copied to --output (sort {sort_secs:.2}s)");
        return Ok(None);
    }

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
    Ok(Some(count))
}
