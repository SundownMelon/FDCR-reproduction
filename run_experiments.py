#!/usr/bin/env python
"""
Batch Experiment Runner for FDCR Reproduction

This script runs all combinations of experiment configurations:
- Attack patterns: base_backdoor, dba_backdoor
- Alpha (Non-IID) values: 0.9, 0.1
- Multiple random seeds for reproducibility

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import os
import sys
import argparse
import subprocess
import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    attack_pattern: str  # 'base_backdoor' or 'dba_backdoor'
    alpha: float  # Dirichlet parameter (0.9 or 0.1)
    seed: int  # Random seed for reproducibility
    dataset: str  # Dataset name (default: fl_cifar10)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    steady_state_acc: float
    steady_state_asr: float
    mean_filtered_ratio: float
    detection_accuracy: float
    success: bool
    error_message: Optional[str] = None


# Default experiment configurations
DEFAULT_ATTACK_PATTERNS = ['base_backdoor', 'dba_backdoor']
DEFAULT_ALPHA_VALUES = [0.9, 0.1]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_DATASET = 'fl_cifar10'


def generate_experiment_configs(
    attack_patterns: List[str] = None,
    alpha_values: List[float] = None,
    seeds: List[int] = None,
    dataset: str = None
) -> List[ExperimentConfig]:
    """
    Generate all combinations of experiment configurations.
    
    Args:
        attack_patterns: List of attack patterns to test
        alpha_values: List of Dirichlet alpha values
        seeds: List of random seeds
        dataset: Dataset name
    
    Returns:
        List of ExperimentConfig objects
    
    Requirements: 5.1, 5.2
    """
    if attack_patterns is None:
        attack_patterns = DEFAULT_ATTACK_PATTERNS
    if alpha_values is None:
        alpha_values = DEFAULT_ALPHA_VALUES
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if dataset is None:
        dataset = DEFAULT_DATASET
    
    configs = []
    for attack in attack_patterns:
        for alpha in alpha_values:
            for seed in seeds:
                configs.append(ExperimentConfig(
                    attack_pattern=attack,
                    alpha=alpha,
                    seed=seed,
                    dataset=dataset
                ))
    
    return configs


def build_command(config: ExperimentConfig, device_id: int = 0) -> List[str]:
    """
    Build the command line arguments for running a single experiment.
    
    Args:
        config: Experiment configuration
        device_id: GPU device ID
    
    Returns:
        List of command line arguments
    """
    # Generate a unique csv_name for this experiment
    csv_name = f"{config.attack_pattern}_alpha{config.alpha}_seed{config.seed}"
    
    cmd = [
        sys.executable, 'main.py',
        '--device_id', str(device_id),
        '--task', 'label_skew',
        '--dataset', config.dataset,
        '--attack_type', 'backdoor',
        '--optim', 'FedFish',
        '--server', 'OurRandomControl',
        '--seed', str(config.seed),
        '--csv_log',
        '--csv_name', csv_name,
        # Override config options (yacs requires key value separated)
        'DATASET.beta', str(config.alpha),
        'attack.backdoor.evils', config.attack_pattern,
    ]
    
    return cmd


def run_single_experiment(
    config: ExperimentConfig,
    device_id: int = 0,
    dry_run: bool = False
) -> ExperimentResult:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        device_id: GPU device ID
        dry_run: If True, only print the command without executing
    
    Returns:
        ExperimentResult object
    
    Requirements: 5.2
    """
    cmd = build_command(config, device_id)
    
    print(f"\n{'='*60}")
    print(f"Running experiment:")
    print(f"  Attack: {config.attack_pattern}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Seed: {config.seed}")
    print(f"  Dataset: {config.dataset}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return ExperimentResult(
            config=config,
            steady_state_acc=0.0,
            steady_state_asr=0.0,
            mean_filtered_ratio=0.0,
            detection_accuracy=0.0,
            success=True,
            error_message="Dry run - no execution"
        )
    
    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        
        if result.returncode != 0:
            print(f"Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return ExperimentResult(
                config=config,
                steady_state_acc=0.0,
                steady_state_asr=0.0,
                mean_filtered_ratio=0.0,
                detection_accuracy=0.0,
                success=False,
                error_message=result.stderr[:500] if result.stderr else "Unknown error"
            )
        
        # Parse results from the summary file
        summary = parse_experiment_summary(config)
        
        return ExperimentResult(
            config=config,
            steady_state_acc=summary.get('steady_state_acc', 0.0),
            steady_state_asr=summary.get('steady_state_asr', 0.0),
            mean_filtered_ratio=summary.get('mean_filtered_ratio', 0.0),
            detection_accuracy=summary.get('detection_accuracy', 0.0),
            success=True
        )
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return ExperimentResult(
            config=config,
            steady_state_acc=0.0,
            steady_state_asr=0.0,
            mean_filtered_ratio=0.0,
            detection_accuracy=0.0,
            success=False,
            error_message="Experiment timed out after 2 hours"
        )
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        return ExperimentResult(
            config=config,
            steady_state_acc=0.0,
            steady_state_asr=0.0,
            mean_filtered_ratio=0.0,
            detection_accuracy=0.0,
            success=False,
            error_message=str(e)
        )


def find_experiment_result_path(config: ExperimentConfig) -> Optional[Path]:
    """
    Find the result directory for a given experiment configuration.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Path to the result directory, or None if not found
    """
    csv_name = f"{config.attack_pattern}_alpha{config.alpha}_seed{config.seed}"
    
    # Construct the expected path based on CsvWriter.model_folder_path logic
    # Path: ./data/label_skew/{attack_pattern}/{bad_client_rate}/{dataset}/{alpha}/{server}/{optim}/{csv_name}
    base_path = Path('./data/label_skew')
    result_path = base_path / config.attack_pattern / '0.3' / config.dataset / str(config.alpha) / 'OurRandomControl' / 'FedFish' / csv_name
    
    if result_path.exists():
        return result_path
    
    # Try alternative path structure (without bad_client_rate in path)
    alt_path = base_path / config.attack_pattern / config.dataset / str(config.alpha) / 'OurRandomControl' / 'FedFish' / csv_name
    if alt_path.exists():
        return alt_path
    
    return None


def parse_experiment_summary(config: ExperimentConfig) -> Dict[str, float]:
    """
    Parse the experiment summary file for a given configuration.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dictionary with summary metrics
    """
    result_path = find_experiment_result_path(config)
    
    if result_path is None:
        print(f"Warning: Could not find result directory for {config}")
        return {}
    
    summary_file = result_path / 'experiment_summary.csv'
    
    if not summary_file.exists():
        print(f"Warning: Summary file not found at {summary_file}")
        return {}
    
    summary = {}
    try:
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get('metric', '')
                value = row.get('value', '0')
                try:
                    summary[metric] = float(value)
                except ValueError:
                    summary[metric] = 0.0
    except Exception as e:
        print(f"Error parsing summary file: {e}")
    
    return summary


def run_all_experiments(
    attack_patterns: List[str] = None,
    alpha_values: List[float] = None,
    seeds: List[int] = None,
    dataset: str = None,
    device_id: int = 0,
    dry_run: bool = False
) -> List[ExperimentResult]:
    """
    Run all experiment configuration combinations.
    
    Args:
        attack_patterns: List of attack patterns to test
        alpha_values: List of Dirichlet alpha values
        seeds: List of random seeds
        dataset: Dataset name
        device_id: GPU device ID
        dry_run: If True, only print commands without executing
    
    Returns:
        List of ExperimentResult objects
    
    Requirements: 5.1, 5.2, 5.4
    """
    configs = generate_experiment_configs(
        attack_patterns=attack_patterns,
        alpha_values=alpha_values,
        seeds=seeds,
        dataset=dataset
    )
    
    print(f"\n{'#'*60}")
    print(f"FDCR Reproduction Experiments")
    print(f"{'#'*60}")
    print(f"Total experiments to run: {len(configs)}")
    print(f"Attack patterns: {attack_patterns or DEFAULT_ATTACK_PATTERNS}")
    print(f"Alpha values: {alpha_values or DEFAULT_ALPHA_VALUES}")
    print(f"Seeds: {seeds or DEFAULT_SEEDS}")
    print(f"Dataset: {dataset or DEFAULT_DATASET}")
    print(f"{'#'*60}\n")
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Starting experiment...")
        result = run_single_experiment(config, device_id, dry_run)
        results.append(result)
        
        if result.success:
            print(f"✓ Experiment completed successfully")
            print(f"  ACC: {result.steady_state_acc:.2f}%")
            print(f"  ASR: {result.steady_state_asr:.2f}%")
            print(f"  Filtered Ratio: {result.mean_filtered_ratio:.4f}")
        else:
            print(f"✗ Experiment failed: {result.error_message}")
    
    return results




def aggregate_results_by_config(results: List[ExperimentResult]) -> Dict[Tuple[str, float], Dict[str, List[float]]]:
    """
    Aggregate results by (attack_pattern, alpha) configuration across seeds.
    
    Args:
        results: List of experiment results
    
    Returns:
        Dictionary mapping (attack, alpha) to lists of metrics
    """
    aggregated = {}
    
    for result in results:
        if not result.success:
            continue
        
        key = (result.config.attack_pattern, result.config.alpha)
        
        if key not in aggregated:
            aggregated[key] = {
                'acc': [],
                'asr': [],
                'filtered_ratio': []
            }
        
        aggregated[key]['acc'].append(result.steady_state_acc)
        aggregated[key]['asr'].append(result.steady_state_asr)
        aggregated[key]['filtered_ratio'].append(result.mean_filtered_ratio)
    
    return aggregated


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation of a list of values."""
    if not values:
        return 0.0, 0.0
    
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    
    return mean, std


def generate_summary_table(
    results: List[ExperimentResult] = None,
    results_dir: str = None,
    output_file: str = 'experiment_summary.md'
) -> str:
    """
    Generate a markdown summary table from experiment results.
    
    Args:
        results: List of ExperimentResult objects (if available)
        results_dir: Directory containing experiment results (alternative to results)
        output_file: Output file path for the markdown table
    
    Returns:
        Markdown table as string
    
    Requirements: 5.3
    """
    if results is None and results_dir is not None:
        # Parse results from directory
        results = parse_results_from_directory(results_dir)
    
    if results is None or len(results) == 0:
        return "No results available to generate summary table."
    
    # Aggregate results by configuration
    aggregated = aggregate_results_by_config(results)
    
    # Build markdown table
    lines = []
    lines.append("# FDCR Reproduction Experiment Results")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Attack Pattern | Alpha (Non-IID) | ACC (%) | ASR (%) | Filtered Ratio |")
    lines.append("|----------------|-----------------|---------|---------|----------------|")
    
    # Sort by attack pattern and alpha for consistent ordering
    for (attack, alpha) in sorted(aggregated.keys()):
        metrics = aggregated[(attack, alpha)]
        
        acc_mean, acc_std = compute_mean_std(metrics['acc'])
        asr_mean, asr_std = compute_mean_std(metrics['asr'])
        fr_mean, fr_std = compute_mean_std(metrics['filtered_ratio'])
        
        # Format with mean ± std
        acc_str = f"{acc_mean:.2f} ± {acc_std:.2f}" if acc_std > 0 else f"{acc_mean:.2f}"
        asr_str = f"{asr_mean:.2f} ± {asr_std:.2f}" if asr_std > 0 else f"{asr_mean:.2f}"
        fr_str = f"{fr_mean:.4f} ± {fr_std:.4f}" if fr_std > 0 else f"{fr_mean:.4f}"
        
        lines.append(f"| {attack} | {alpha} | {acc_str} | {asr_str} | {fr_str} |")
    
    lines.append("")
    lines.append("## Experiment Details")
    lines.append("")
    lines.append("### Configuration")
    lines.append("- Optimizer: FedFish")
    lines.append("- Server: OurRandomControl (FDCR)")
    lines.append("- Task: label_skew")
    lines.append("")
    
    # Add individual experiment results
    lines.append("### Individual Runs")
    lines.append("")
    lines.append("| Attack | Alpha | Seed | ACC (%) | ASR (%) | Filtered Ratio | Status |")
    lines.append("|--------|-------|------|---------|---------|----------------|--------|")
    
    for result in sorted(results, key=lambda r: (r.config.attack_pattern, r.config.alpha, r.config.seed)):
        status = "✓" if result.success else "✗"
        if result.success:
            lines.append(
                f"| {result.config.attack_pattern} | {result.config.alpha} | {result.config.seed} | "
                f"{result.steady_state_acc:.2f} | {result.steady_state_asr:.2f} | "
                f"{result.mean_filtered_ratio:.4f} | {status} |"
            )
        else:
            error_short = result.error_message[:30] if result.error_message else "Unknown"
            lines.append(
                f"| {result.config.attack_pattern} | {result.config.alpha} | {result.config.seed} | "
                f"- | - | - | {status} ({error_short}...) |"
            )
    
    markdown_content = '\n'.join(lines)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"\nSummary table written to: {output_file}")
    
    return markdown_content


def parse_results_from_directory(results_dir: str) -> List[ExperimentResult]:
    """
    Parse experiment results from a results directory.
    
    Args:
        results_dir: Base directory containing experiment results
    
    Returns:
        List of ExperimentResult objects
    """
    results = []
    base_path = Path(results_dir)
    
    if not base_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    # Walk through the directory structure to find experiment_summary.csv files
    for summary_file in base_path.rglob('experiment_summary.csv'):
        try:
            # Parse the path to extract configuration
            # Expected path: .../attack_pattern/bad_rate/dataset/alpha/server/optim/csv_name/
            parts = summary_file.parent.parts
            
            # Find csv_name which contains attack_alpha_seed pattern
            csv_name = parts[-1]
            
            # Parse csv_name: {attack}_alpha{alpha}_seed{seed}
            if '_alpha' in csv_name and '_seed' in csv_name:
                attack_part = csv_name.split('_alpha')[0]
                alpha_seed_part = csv_name.split('_alpha')[1]
                alpha_str, seed_str = alpha_seed_part.split('_seed')
                
                config = ExperimentConfig(
                    attack_pattern=attack_part,
                    alpha=float(alpha_str),
                    seed=int(seed_str),
                    dataset='fl_cifar10'  # Default, could be parsed from path
                )
                
                # Parse the summary file
                summary = {}
                with open(summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metric = row.get('metric', '')
                        value = row.get('value', '0')
                        try:
                            summary[metric] = float(value)
                        except ValueError:
                            summary[metric] = 0.0
                
                results.append(ExperimentResult(
                    config=config,
                    steady_state_acc=summary.get('steady_state_acc', 0.0),
                    steady_state_asr=summary.get('steady_state_asr', 0.0),
                    mean_filtered_ratio=summary.get('mean_filtered_ratio', 0.0),
                    detection_accuracy=summary.get('detection_accuracy', 0.0),
                    success=True
                ))
                
        except Exception as e:
            print(f"Error parsing {summary_file}: {e}")
            continue
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FDCR Reproduction Batch Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default experiments
  python run_experiments.py
  
  # Run with specific attack patterns
  python run_experiments.py --attacks base_backdoor dba_backdoor
  
  # Run with specific alpha values
  python run_experiments.py --alphas 0.9 0.1
  
  # Run with specific seeds
  python run_experiments.py --seeds 0 1 2 3 4
  
  # Dry run (print commands without executing)
  python run_experiments.py --dry-run
  
  # Generate summary from existing results
  python run_experiments.py --summary-only --results-dir ./data
  
  # Specify dataset
  python run_experiments.py --dataset fl_cifar100
        """
    )
    
    parser.add_argument(
        '--attacks', 
        nargs='+', 
        default=DEFAULT_ATTACK_PATTERNS,
        choices=['base_backdoor', 'dba_backdoor'],
        help='Attack patterns to test (default: base_backdoor dba_backdoor)'
    )
    
    parser.add_argument(
        '--alphas', 
        nargs='+', 
        type=float, 
        default=DEFAULT_ALPHA_VALUES,
        help='Dirichlet alpha values for Non-IID (default: 0.9 0.1)'
    )
    
    parser.add_argument(
        '--seeds', 
        nargs='+', 
        type=int, 
        default=DEFAULT_SEEDS,
        help='Random seeds for reproducibility (default: 0 1 2)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default=DEFAULT_DATASET,
        help='Dataset name (default: fl_cifar10)'
    )
    
    parser.add_argument(
        '--device-id', 
        type=int, 
        default=0,
        help='GPU device ID (default: 0)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Print commands without executing'
    )
    
    parser.add_argument(
        '--summary-only', 
        action='store_true',
        help='Only generate summary from existing results'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='./data',
        help='Directory containing experiment results (default: ./data)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='experiment_summary.md',
        help='Output file for summary table (default: experiment_summary.md)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the batch experiment runner."""
    args = parse_args()
    
    if args.summary_only:
        # Only generate summary from existing results
        print("Generating summary from existing results...")
        results = parse_results_from_directory(args.results_dir)
        if results:
            summary = generate_summary_table(results=results, output_file=args.output)
            print(summary)
        else:
            print("No results found in the specified directory.")
        return
    
    # Run all experiments
    results = run_all_experiments(
        attack_patterns=args.attacks,
        alpha_values=args.alphas,
        seeds=args.seeds,
        dataset=args.dataset,
        device_id=args.device_id,
        dry_run=args.dry_run
    )
    
    # Generate summary table
    if not args.dry_run:
        print("\n" + "="*60)
        print("Generating summary table...")
        print("="*60)
        summary = generate_summary_table(results=results, output_file=args.output)
        print(summary)
    
    # Print final status
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
