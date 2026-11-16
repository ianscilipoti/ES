//! Print the settings the resource-efficient planner would choose, one per line.
//!
//! The planner also needs the dataset size: pass either a GenSort input file
//! (`--input`, size read from disk) or the size directly (`--dataset-mb`).
//!
//! Example:
//!   print_plan --memory-mb 2048 --max-threads 16 --input datasets/g1.data
//!   print_plan --memory-mb 2048 --max-threads 16 --dataset-mb 204800
//!
//! Pass `--write-output true` to reproduce what `sort_to_file --use-planner
//! --write-output true` would pick (it reserves its output write buffer out
//! of the budget before planning).

use clap::Parser;
use es::GenSortInputDirect;
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::sort_policy_sub::{PlannerConfig, PlannerRegime, plan_resource_efficient};
use std::path::PathBuf;

/// Must match `WRITE_BUFFER_SIZE` in sort_to_file.rs.
const SORT_TO_FILE_WRITE_BUFFER_MB: f64 = 8.0;

#[derive(Parser)]
#[command(about = "Print the planner's chosen sort settings for a memory / thread budget")]
struct Args {
    /// Total memory budget in MB
    #[arg(long)]
    memory_mb: f64,

    /// Maximum thread count
    #[arg(long)]
    max_threads: usize,

    /// GenSort input file (dataset size is taken from its file size)
    #[arg(short, long, required_unless_present = "dataset_mb", conflicts_with = "dataset_mb")]
    input: Option<PathBuf>,

    /// Dataset size in MB (alternative to --input)
    #[arg(long)]
    dataset_mb: Option<f64>,

    /// Fraction of memory reserved for sparse-index pages
    #[arg(long, default_value = "0.05")]
    sparse_index_fraction: f64,

    /// Reserve sort_to_file's output write buffer from the budget (`true`/`false`)
    #[arg(long, default_value = "false", action = clap::ArgAction::Set)]
    write_output: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let dataset_mb = match (&args.input, args.dataset_mb) {
        (Some(path), _) => GenSortInputDirect::new(path)?.file_size()? as f64 / (1024.0 * 1024.0),
        (None, Some(mb)) => mb,
        (None, None) => unreachable!("clap requires --input or --dataset-mb"),
    };

    let write_reserve_mb = if args.write_output {
        SORT_TO_FILE_WRITE_BUFFER_MB
    } else {
        0.0
    };
    let planner_memory_mb = args.memory_mb - write_reserve_mb;
    if planner_memory_mb <= 0.0 {
        return Err(format!(
            "--memory-mb {} leaves no memory after the {write_reserve_mb} MB write reserve",
            args.memory_mb
        )
        .into());
    }

    let plan = plan_resource_efficient(&PlannerConfig {
        dataset_mb,
        memory_mb: planner_memory_mb,
        max_threads: args.max_threads,
        page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
        sparse_index_fraction: args.sparse_index_fraction,
        ..PlannerConfig::default()
    });

    let regime = match plan.regime {
        PlannerRegime::ThreadBound => "thread-bound",
        PlannerRegime::MemoryBound => "memory-bound",
    };

    println!("== Inputs ==");
    println!("Dataset size (MB):          {dataset_mb:.2}");
    println!("Memory budget (MB):         {:.2}", args.memory_mb);
    println!("Write reserve (MB):         {write_reserve_mb:.2}");
    println!("Memory given to planner:    {planner_memory_mb:.2}");
    println!("Max threads:                {}", args.max_threads);
    println!("Sparse index fraction:      {}", args.sparse_index_fraction);
    println!("Page size (KB):             {}", DEFAULT_BUFFER_SIZE / 1024);
    println!();
    println!("== Planner settings ==");
    println!("Regime:                     {regime}");
    println!("Run-gen threads (T_gen):    {}", plan.run_gen_threads);
    println!("Merge threads (T_merge):    {}", plan.merge_threads);
    println!("RG buffer per thread (MB):  {:.2}", plan.rg_buf_mb);
    println!("Merge fan-in:               {}", plan.merge_fanin);
    println!();
    println!("== Derived estimates ==");
    println!("Estimated runs:             {}", plan.num_runs);
    println!("Run-gen memory (MB):        {:.2}", plan.run_gen_memory_mb);
    println!("Merge memory (MB):          {:.2}", plan.merge_memory_mb);
    println!("Single merge step:          {}", plan.is_single_step);
    println!();
    // rg_buf_mb uses `{}` (shortest round-trip form) so pasting the flags
    // reproduces the planner's buffer size exactly.
    println!("== sort_to_file flags (copy/paste) ==");
    println!(
        "--run-gen-threads {} --merge-threads {} --rg-buf-mb {} --merge-fanin {} \
         --sparse-index-fraction {} --write-output {}",
        plan.run_gen_threads,
        plan.merge_threads,
        plan.rg_buf_mb,
        plan.merge_fanin,
        args.sparse_index_fraction,
        args.write_output,
    );

    Ok(())
}
