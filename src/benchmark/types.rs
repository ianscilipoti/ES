use crate::SortStats;
use crate::sort::core::engine::PartitionType;
use std::path::PathBuf;

#[derive(Clone)]
pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub stats: Vec<SortStats>,
}

/// Which run-file format the sorter uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunFormatChoice {
    /// Length-prefixed records (`[key_len][value_len][key][value]`).
    Plain,
    /// Offset-value coded records.
    Ovc,
    /// Fixed-size records with no headers (`[key][value]`). Supports packed
    /// output via `BenchmarkConfig::output`.
    Fixed { key_size: usize, value_size: usize },
}

impl RunFormatChoice {
    /// The format the older `--ovc <bool>` flag selects.
    pub fn from_ovc_flag(use_ovc: bool) -> Self {
        if use_ovc { Self::Ovc } else { Self::Plain }
    }
}

impl std::fmt::Display for RunFormatChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plain => write!(f, "plain"),
            Self::Ovc => write!(f, "ovc"),
            Self::Fixed {
                key_size,
                value_size,
            } => write!(f, "fixed ({key_size}-byte keys, {value_size}-byte values)"),
        }
    }
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub config_name: String,
    pub warmup_runs: usize,
    pub benchmark_runs: usize,
    pub cooldown_seconds: u64,
    pub verify: bool,
    pub temp_dir: PathBuf,
    pub run_gen_threads: usize,
    pub run_format: RunFormatChoice,
    pub rg_buf_mb: f64,
    pub run_gen_memory_mb: f64,
    pub merge_threads: usize,
    pub merge_fanin: usize,
    pub merge_memory_mb: f64,
    pub imbalance_factor: f64,
    pub partition_type: PartitionType,
    pub discard_final_output: bool,
    /// Write the sorted result to this file as packed fixed-size records.
    /// Requires `RunFormatChoice::Fixed`. The file is kept after the run
    /// (each benchmark run overwrites it).
    pub output: Option<PathBuf>,
    /// Fraction of run-gen / merge memory reserved for sparse-index pages.
    /// Default 0.05; pass through to `SorterCore::set_sparse_index_fraction`.
    pub sparse_index_fraction: f64,
}
