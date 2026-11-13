use super::input::BenchmarkInputProvider;
use super::types::{BenchmarkConfig, BenchmarkResult, RunFormatChoice};
use super::verification::OutputVerifier;
use crate::sort::core::engine::{SortHooks, SorterCore};
use crate::sort::fixed::sorter::ExternalSorterFixed;
use crate::{
    ExternalSorter, ExternalSorterWithOVC, SortInput, SortOutput, SortStats, Sorter,
    dispatch_fixed_layout,
};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

/// Apply the settings shared by every run format.
fn configure_sorter<H: SortHooks>(sorter: &mut SorterCore<H>, config: &BenchmarkConfig) {
    sorter.set_imbalance_factor(config.imbalance_factor);
    sorter.set_partition_type(config.partition_type);
    sorter.set_discard_final_output(config.discard_final_output);
    sorter.set_sparse_index_fraction(config.sparse_index_fraction);
    if let Some(output) = &config.output {
        // Variable-size formats reject this with an error from `sort`.
        sorter.set_packed_output(output);
    }
}

fn sort_fixed<const K: usize, const V: usize>(
    config: &BenchmarkConfig,
    rg_buf_bytes: usize,
    input: Box<dyn SortInput>,
    temp_dir: &Path,
) -> Result<Box<dyn SortOutput>, String> {
    let mut sorter = ExternalSorterFixed::<K, V>::new(
        config.run_gen_threads,
        rg_buf_bytes,
        config.merge_threads,
        config.merge_fanin,
        temp_dir,
    );
    configure_sorter(&mut sorter, config);
    sorter.sort(input)
}

pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    input_provider: Box<dyn BenchmarkInputProvider>,
    verifier: Option<Box<dyn OutputVerifier>>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig, input_provider: Box<dyn BenchmarkInputProvider>) -> Self {
        Self {
            config,
            input_provider,
            verifier: None,
        }
    }

    pub fn set_verifier(&mut self, verifier: Box<dyn OutputVerifier>) {
        self.verifier = Some(verifier);
    }

    /// New single-configuration benchmark entrypoint
    /// Runs warmups (if configured) and the requested number of runs using the provided params
    pub fn run_configuration(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let dataset_mb = self.input_provider.estimate_data_size_mb()?;
        self.print_benchmark_header(dataset_mb)?;

        println!("Running benchmark for config: {}", self.config.config_name);
        println!(
            "Parameters: Run Gen Threads: {}, Run format: {}, RG Buffer: {:.2} MB, Run Gen Memory: {:.1} MB, Merge Threads: {}, Merge Fanin: {}, Merge Memory: {:.1} MB, Imbalance Factor: {:.1}, Partition: {:?}",
            self.config.run_gen_threads,
            self.config.run_format,
            self.config.rg_buf_mb,
            self.config.run_gen_memory_mb,
            self.config.merge_threads,
            self.config.merge_fanin,
            self.config.merge_memory_mb,
            self.config.imbalance_factor,
            self.config.partition_type,
        );
        println!("{}", "=".repeat(80));

        if self.config.warmup_runs > 0 {
            self.run_warmup_runs()?;
        }

        let per_run_stats = self.run_policy_benchmark()?;
        Ok(BenchmarkResult {
            config: self.config.clone(),
            stats: per_run_stats,
        })
    }

    fn print_benchmark_header(&self, dataset_mb: f64) -> Result<(), Box<dyn std::error::Error>> {
        let total_entries = self.input_provider.get_entry_count();

        println!("\n=== BENCHMARK MODE ===");
        println!("{}", self.input_provider.get_description());
        if let Some(entries) = total_entries {
            println!("Total entries: {}", entries);
        }
        println!("Estimated data size: {:.2} MB", dataset_mb);
        println!("Run generation threads: {}", self.config.run_gen_threads);
        println!("Run format: {}", self.config.run_format);
        if let Some(output) = &self.config.output {
            println!("Sorted output file: {:?}", output);
        }
        println!("RG buffer (MB): {:.2}", self.config.rg_buf_mb);
        println!(
            "Run generation memory (MB): {:.1}",
            self.config.run_gen_memory_mb
        );
        println!("Merge threads: {}", self.config.merge_threads);
        println!("Merge fan-in: {}", self.config.merge_fanin);
        println!("Merge memory (MB): {:.1}", self.config.merge_memory_mb);
        println!("Partition type: {:?}", self.config.partition_type);
        println!("Temporary directory: {:?}", self.config.temp_dir);
        println!("Warmup runs: {}", self.config.warmup_runs);
        println!("Runs per configuration: {}", self.config.benchmark_runs);
        println!(
            "Cooldown between runs (s): {}",
            self.config.cooldown_seconds
        );
        println!("Verify output: {}", self.config.verify);
        println!();

        Ok(())
    }

    fn run_warmup_runs(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("  Performing {} warmup run(s)...", self.config.warmup_runs);

        for warmup in 1..=self.config.warmup_runs {
            print!("    Warmup {}/{}: ", warmup, self.config.warmup_runs);

            let temp_dir = self.config.temp_dir.join(format!(
                "warmup_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
            ));
            std::fs::create_dir_all(&temp_dir)?;

            let input = self.input_provider.create_sort_input()?;

            let output = self.run_single_sort(input, &temp_dir)?;
            let stats = output.stats();
            let run_gen_stats = stats.run_gen_stats;
            let multi_merge_stats = stats.per_merge_stats;

            let total_merge_time_ms: u128 = multi_merge_stats.iter().map(|s| s.time_ms).sum();

            println!(
                "{:.2}s",
                run_gen_stats.time_ms as f64 / 1000.0 + total_merge_time_ms as f64 / 1000.0
            );

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            self.sync_filesystem()?;

            // Cooldown between warmup runs (except after the last warmup)
            if self.config.cooldown_seconds > 0 && warmup < self.config.warmup_runs {
                println!(
                    "    Cooling down for {}s before next warmup...",
                    self.config.cooldown_seconds
                );
                std::thread::sleep(std::time::Duration::from_secs(self.config.cooldown_seconds));
            }
        }

        println!("  Warmup complete.\n");
        Ok(())
    }

    fn run_policy_benchmark(&self) -> Result<Vec<SortStats>, Box<dyn std::error::Error>> {
        let mut per_run_stats: Vec<SortStats> = Vec::new();

        for run in 1..=self.config.benchmark_runs {
            print!("  Run {}/{}: ", run, self.config.benchmark_runs);

            let temp_dir = self.config.temp_dir.join(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
                    .to_string(),
            );
            std::fs::create_dir_all(&temp_dir)?;

            let input = self.input_provider.create_sort_input()?;

            let output = self.run_single_sort(input, &temp_dir)?;

            println!("{}", output.stats());

            // Collect per-run SortStats and store
            let stats_this_run = output.stats();
            let run_gen_time_ms = stats_this_run.run_gen_stats.time_ms;
            let merge_time_ms: u128 = stats_this_run
                .per_merge_stats
                .iter()
                .map(|m| m.time_ms)
                .sum();
            per_run_stats.push(stats_this_run.clone());
            println!(
                "{:.2}s",
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0
            );

            // Verify if requested (only on first run)
            if self.config.verify && run == 1 {
                if let Some(ref verifier) = self.verifier {
                    println!("    Verifying sorted output...");
                    verifier.verify(&*output)?;
                    println!("    Verification passed!");
                } else {
                    println!("    Warning: Verification requested but no verifier provided");
                }
            }

            drop(output); // Release resources
            println!();

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            self.sync_filesystem()?;

            // Cooldown between benchmark runs (except after the last run)
            if self.config.cooldown_seconds > 0 && run < self.config.benchmark_runs {
                println!(
                    "  Cooling down for {}s before next run...",
                    self.config.cooldown_seconds
                );
                std::thread::sleep(std::time::Duration::from_secs(self.config.cooldown_seconds));
            }
        }

        Ok(per_run_stats)
    }

    fn run_single_sort(
        &self,
        input: Box<dyn SortInput>,
        temp_dir: &Path,
    ) -> Result<Box<dyn SortOutput>, Box<dyn std::error::Error>> {
        let config = &self.config;
        let rg_buf_bytes = (config.rg_buf_mb * 1024.0 * 1024.0) as usize;
        let output = match config.run_format {
            RunFormatChoice::Ovc => {
                let mut sorter = ExternalSorterWithOVC::new(
                    config.run_gen_threads,
                    rg_buf_bytes,
                    config.merge_threads,
                    config.merge_fanin,
                    temp_dir,
                );
                configure_sorter(&mut sorter, config);
                sorter.sort(input)?
            }
            RunFormatChoice::Plain => {
                let mut sorter = ExternalSorter::new(
                    config.run_gen_threads,
                    rg_buf_bytes,
                    config.merge_threads,
                    config.merge_fanin,
                    temp_dir,
                );
                configure_sorter(&mut sorter, config);
                sorter.sort(input)?
            }
            RunFormatChoice::Fixed {
                key_size,
                value_size,
            } => dispatch_fixed_layout!(
                key_size,
                value_size,
                sort_fixed(config, rg_buf_bytes, input, temp_dir)
            )??,
        };

        Ok(output)
    }

    fn sync_filesystem(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let dir_fd = File::open(&self.config.temp_dir).map_err(|e| {
                format!("Failed to open directory {:?}: {}", self.config.temp_dir, e)
            })?;

            #[cfg(target_os = "linux")]
            {
                libc::syncfs(dir_fd.as_raw_fd());
            }

            #[cfg(any(target_os = "macos", target_os = "freebsd"))]
            {
                libc::fsync(dir_fd.as_raw_fd());
            }
        }
        Ok(())
    }
}
