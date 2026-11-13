use clap::{ArgAction, Parser};
use es::SUPPORTED_FIXED_LAYOUTS;
use es::benchmark::{
    BenchmarkConfig, BenchmarkInputProvider, BenchmarkResult, BenchmarkRunner,
    GenSortInputProvider, RunFormatChoice, SimpleVerifier, print_benchmark_summary,
};
use es::diskio::constants::DEFAULT_BUFFER_SIZE;
use es::sort::core::engine::PartitionType;
use es::sort_policy_sub::{PlannerConfig, plan_resource_efficient};
use std::path::PathBuf;

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

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum RunFormatArg {
    /// Length-prefixed records
    Plain,
    /// Offset-value coding
    Ovc,
    /// Fixed-size records sized by --key-size / --value-size; supports --output
    Fixed,
}

#[derive(Parser)]
struct SortArgs {
    /// Configuration name for labeling the run
    #[arg(short, default_value = "no_name")]
    name: String,
    /// Input GenSort file path
    #[arg(short, long)]
    input: PathBuf,

    /// Key size in bytes for each fixed-size record (GenSort default: 10)
    #[arg(long, default_value = "10")]
    key_size: usize,

    /// Value size in bytes for each fixed-size record (GenSort default: 90)
    #[arg(long, default_value = "90")]
    value_size: usize,

    /// Directory for temporary files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Verify sorted output
    #[arg(short, long)]
    verify: bool,

    /// Number of benchmark runs per configuration
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,

    /// Number of warmup runs before benchmarking (not included in results)
    #[arg(long, default_value = "0")]
    warmup_runs: usize,

    /// Cooldown seconds between runs
    #[arg(long, default_value = "0")]
    cooldown_seconds: u64,

    /// Use OVC (Offset Value Coding) format. Ignored when --run-format is given.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    ovc: bool,

    /// Run-file format (`plain`, `ovc`, `fixed`). Defaults to the --ovc setting.
    #[arg(long, value_name = "FORMAT")]
    run_format: Option<RunFormatArg>,

    /// Write the sorted records to this file in the input's packed layout
    /// (key bytes then value bytes, no delimiters). Requires --run-format fixed.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Automatically derive run_gen_threads, merge_threads, rg_buf_mb and
    /// merge_fanin from the dataset size and the budget (--memory-mb,
    /// --max-threads).  The planner applies the two-regime resource-efficient
    /// policy described in the paper.  When set, the four manual tuning args
    /// below become optional.
    #[arg(long, default_value = "false")]
    use_planner: bool,

    /// Available memory budget in MB (used by --use-planner / --print-plan)
    #[arg(long)]
    memory_mb: Option<f64>,

    /// Maximum thread count (used by --use-planner / --print-plan)
    #[arg(long)]
    max_threads: Option<usize>,

    /// Threads for run generation
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    run_gen_threads: Option<usize>,

    /// Threads for merge phase
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    merge_threads: Option<usize>,

    /// RG buffer for run generation (MB)
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    rg_buf_mb: Option<f64>,

    /// Merge fan-in (global per merge operation)
    #[arg(long, required_unless_present_any = ["estimate_size", "use_planner", "print_plan"])]
    merge_fanin: Option<usize>,

    /// Merge imbalance factor (>= 1.0)
    #[arg(long, default_value = "1.0")]
    imbalance_factor: f64,

    /// Merge partition type (`key-only`, `count-balanced`, `size-balanced`)
    #[arg(long, default_value = "size-balanced", value_name = "PARTITION")]
    partition_type: PartitionArg,

    /// Discard final output (no write) for benchmarking
    #[arg(long, default_value = "false")]
    discard_final_output: bool,

    /// Fraction of run-gen / merge memory reserved for sparse-index pages.
    /// The remainder is the data buffer.  Default 0.05 (= 5%).
    #[arg(long, default_value = "0.05")]
    sparse_index_fraction: f64,

    /// Only estimate dataset size (MB) and exit
    #[arg(long, default_value = "false")]
    estimate_size: bool,

    /// Compute and print the planner's resource configuration, then exit
    /// without running the sort.  Requires --memory-mb and --max-threads.
    #[arg(long, default_value = "false")]
    print_plan: bool,
}

/// Resolve the run format and check the output options before doing any work.
fn resolve_run_format(args: &SortArgs) -> Result<RunFormatChoice, String> {
    let run_format = match args.run_format {
        Some(RunFormatArg::Plain) => RunFormatChoice::Plain,
        Some(RunFormatArg::Ovc) => RunFormatChoice::Ovc,
        Some(RunFormatArg::Fixed) => RunFormatChoice::Fixed {
            key_size: args.key_size,
            value_size: args.value_size,
        },
        None => RunFormatChoice::from_ovc_flag(args.ovc),
    };

    if matches!(run_format, RunFormatChoice::Fixed { .. })
        && !SUPPORTED_FIXED_LAYOUTS.contains(&(args.key_size, args.value_size))
    {
        let supported: Vec<String> = SUPPORTED_FIXED_LAYOUTS
            .iter()
            .map(|(k, v)| format!("{k}/{v}"))
            .collect();
        return Err(format!(
            "--run-format fixed is not compiled for {}-byte keys / {}-byte values (supported: {}). \
             Add the size to SUPPORTED_FIXED_LAYOUTS and dispatch_fixed_layout! in src/sort/fixed/sorter.rs",
            args.key_size,
            args.value_size,
            supported.join(", ")
        ));
    }

    if let Some(output) = &args.output {
        if !matches!(run_format, RunFormatChoice::Fixed { .. }) {
            return Err("--output requires --run-format fixed".to_string());
        }
        if args.discard_final_output {
            return Err("--output cannot be combined with --discard-final-output".to_string());
        }
        let same_file = match (
            std::fs::canonicalize(&args.input),
            std::fs::canonicalize(output),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if same_file {
            return Err("--output must not be the input file".to_string());
        }
    }
    Ok(run_format)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = SortArgs::parse();
    let run_format = resolve_run_format(&args)?;
    let output_path = args.output.clone();

    // Create input provider for GenSort files
    let input_provider =
        GenSortInputProvider::with_record_layout(args.input, args.key_size, args.value_size);

    // If only estimating size, compute and print, then exit
    if args.estimate_size {
        let estimated_mb = input_provider.estimate_data_size_mb()?;
        println!("Estimated data size: {:.2} MB", estimated_mb);
        return Ok(());
    }

    // If only printing the plan, compute it and exit without sorting
    if args.print_plan {
        let memory_mb = args
            .memory_mb
            .ok_or("--memory-mb is required when --print-plan is set")?;
        let max_threads = args
            .max_threads
            .ok_or("--max-threads is required when --print-plan is set")?;
        let dataset_mb = input_provider.estimate_data_size_mb()?;
        println!("Estimated data size: {dataset_mb:.2} MB");
        let plan = plan_resource_efficient(&PlannerConfig {
            dataset_mb,
            memory_mb,
            max_threads,
            page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
            ..PlannerConfig::default()
        });
        println!("{plan}");
        return Ok(());
    }

    // Resolve run configuration — either from explicit args or via the planner.
    let (run_gen_threads, merge_threads, rg_buf_mb, merge_fanin) = if args.use_planner {
        let memory_mb = args
            .memory_mb
            .ok_or("--memory-mb is required when --use-planner is set")?;
        let max_threads = args
            .max_threads
            .ok_or("--max-threads is required when --use-planner is set")?;
        let dataset_mb = input_provider.estimate_data_size_mb()?;
        println!("Planner: estimated dataset size = {dataset_mb:.2} MB");
        let plan = plan_resource_efficient(&PlannerConfig {
            dataset_mb,
            memory_mb,
            max_threads,
            page_size_kb: DEFAULT_BUFFER_SIZE as f64 / 1024.0,
            ..PlannerConfig::default()
        });
        println!("Planner: {plan}");
        (
            plan.run_gen_threads,
            plan.merge_threads,
            plan.rg_buf_mb,
            plan.merge_fanin,
        )
    } else {
        let run_gen_threads = args
            .run_gen_threads
            .expect("--run-gen-threads required unless --estimate-size or --use-planner");
        let merge_threads = args
            .merge_threads
            .expect("--merge-threads required unless --estimate-size or --use-planner");
        let rg_buf_mb = args
            .rg_buf_mb
            .expect("--rg-buf-mb required unless --estimate-size or --use-planner");
        let merge_fanin = args
            .merge_fanin
            .expect("--merge-fanin required unless --estimate-size or --use-planner");
        (run_gen_threads, merge_threads, rg_buf_mb, merge_fanin)
    };

    let partition_type: PartitionType = args.partition_type.into();

    // Create benchmark configuration
    let config = BenchmarkConfig {
        config_name: args.name,
        warmup_runs: args.warmup_runs,
        benchmark_runs: args.benchmark_runs,
        cooldown_seconds: args.cooldown_seconds,
        verify: args.verify,
        temp_dir: args.dir,
        run_gen_threads,
        run_format,
        rg_buf_mb,
        run_gen_memory_mb: rg_buf_mb * run_gen_threads as f64,
        merge_threads,
        merge_fanin,
        merge_memory_mb: (merge_fanin as f64)
            * (merge_threads as f64)
            * (DEFAULT_BUFFER_SIZE as f64 / 1024.0 / 1024.0),
        imbalance_factor: args.imbalance_factor,
        partition_type,
        discard_final_output: args.discard_final_output,
        output: output_path,
        sparse_index_fraction: args.sparse_index_fraction,
    };

    let input_provider = Box::new(input_provider);

    // Create benchmark runner
    let mut runner = BenchmarkRunner::new(config, input_provider);

    // Set up verification if requested
    if args.verify {
        let verifier = SimpleVerifier::new(); // With sample output
        runner.set_verifier(Box::new(verifier));
    }

    // Run single benchmark configuration
    let result: BenchmarkResult = runner.run_configuration()?;

    // Print comprehensive summary
    print_benchmark_summary(&result);

    Ok(())
}
