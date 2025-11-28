#!/usr/bin/env python3
"""
Example script demonstrating how to use the throughput parameter generator
for automated benchmarking.

This script shows how to:
1. Generate parameter combinations
2. Run benchmarks with each configuration
3. Collect and analyze results
"""

import json
import subprocess
import sys
import importlib.util
from pathlib import Path

# Import the generator module directly to avoid dependencies
python_dir = Path(__file__).parent.parent / "python"
module_path = python_dir / "sglang" / "throughput_param_generator.py"
spec = importlib.util.spec_from_file_location("throughput_param_generator", module_path)
generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator_module)

ThroughputParamGenerator = generator_module.ThroughputParamGenerator
ParameterDefinition = generator_module.ParameterDefinition

# Benchmark configuration
BENCHMARK_MODULE = "sglang.bench_offline_throughput"


def generate_benchmark_configs(output_file: str, max_configs: int = 10):
    """
    Generate parameter combinations for benchmarking.
    
    Args:
        output_file: Path to save configurations
        max_configs: Maximum number of configurations to generate
    """
    print(f"Generating {max_configs} parameter combinations...")
    
    generator = ThroughputParamGenerator()
    
    # You can customize parameters for your specific use case
    # For example, if you only want to test certain parameters:
    # generator.remove_parameter("attention_backend")
    # generator.add_parameter(ParameterDefinition(
    #     name="custom_param",
    #     values=[1, 2, 3],
    #     description="Custom parameter"
    # ))
    
    combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=max_configs
    )
    
    generator.print_summary(combinations)
    generator.export_to_json(output_file, combinations)
    
    print(f"\nConfigurations saved to: {output_file}")
    return combinations


def build_benchmark_command(config: dict, model_path: str, base_args: dict = None) -> list:
    """
    Build a benchmark command from a parameter configuration.
    
    Args:
        config: Parameter configuration dictionary
        model_path: Path to the model
        base_args: Additional base arguments for the benchmark
        
    Returns:
        Command as a list of strings
    """
    cmd = [
        "python", "-m", BENCHMARK_MODULE,
        "--model-path", model_path,
    ]
    
    # Add base arguments
    if base_args:
        for key, value in base_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Add configuration parameters
    for param_name, param_value in config.items():
        if param_value is not None:  # Skip None values
            flag = f"--{param_name.replace('_', '-')}"
            if isinstance(param_value, bool):
                if param_value:  # Only add flag for True boolean values
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(param_value)])
    
    return cmd


def run_benchmark(config: dict, config_id: int, model_path: str, 
                 result_file: str, dry_run: bool = True):
    """
    Run a single benchmark with the given configuration.
    
    Args:
        config: Parameter configuration
        config_id: Configuration identifier
        model_path: Path to the model
        result_file: Path to save results
        dry_run: If True, only print the command without running it
    """
    base_args = {
        "num_prompts": 100,
        "dataset_name": "random",
        "random_input_len": 1024,
        "random_output_len": 256,
        "result_filename": result_file,
    }
    
    cmd = build_benchmark_command(config, model_path, base_args)
    
    print(f"\n{'='*70}")
    print(f"Configuration {config_id}:")
    print(json.dumps(config, indent=2))
    print(f"\nCommand: {' '.join(cmd)}")
    print('='*70)
    
    if dry_run:
        print("(Dry run - command not executed)")
        return None
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ Configuration {config_id} completed successfully")
        else:
            print(f"✗ Configuration {config_id} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        print(f"✗ Configuration {config_id} timed out")
        return None
    except Exception as e:
        print(f"✗ Configuration {config_id} failed with exception: {e}")
        return None


def analyze_results(result_file: str):
    """
    Analyze benchmark results and find the best configuration.
    
    Args:
        result_file: Path to the results file
    """
    if not Path(result_file).exists():
        print(f"Results file not found: {result_file}")
        return
    
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    if not results:
        print("No results found")
        return
    
    print(f"\n{'='*70}")
    print("Benchmark Results Analysis")
    print('='*70)
    
    # Sort by throughput
    sorted_results = sorted(results, key=lambda x: x.get("total_throughput", 0), reverse=True)
    
    print(f"\nTop 5 configurations by total throughput:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Configuration:")
        print(f"   Total throughput: {result.get('total_throughput', 0):.2f} tokens/s")
        print(f"   Output throughput: {result.get('output_throughput', 0):.2f} tokens/s")
        print(f"   Request throughput: {result.get('request_throughput', 0):.2f} req/s")
        print(f"   Total latency: {result.get('total_latency', 0):.2f} s")
    
    print(f"\n{'='*70}\n")


def main():
    """Main function demonstrating the workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Example script for automated throughput benchmarking"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model path for benchmarking"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=10,
        help="Maximum number of configurations to generate"
    )
    parser.add_argument(
        "--output-configs",
        type=str,
        default="/tmp/benchmark_configs.json",
        help="Output file for configurations"
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="/tmp/benchmark_results.jsonl",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the benchmarks (default is dry-run)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_results(args.result_file)
        return
    
    # Step 1: Generate configurations
    configs = generate_benchmark_configs(args.output_configs, args.max_configs)
    
    # Step 2: Run benchmarks (or dry-run)
    print(f"\n{'='*70}")
    if args.run:
        print("Running benchmarks...")
        print("WARNING: This may take a long time depending on the number of configurations")
    else:
        print("DRY RUN MODE - No benchmarks will actually run")
        print("Use --run flag to execute benchmarks")
    print('='*70)
    
    for i, config in enumerate(configs, 1):
        run_benchmark(
            config=config,
            config_id=i,
            model_path=args.model_path,
            result_file=args.result_file,
            dry_run=not args.run
        )
    
    # Step 3: Analyze results (if we actually ran)
    if args.run:
        analyze_results(args.result_file)


if __name__ == "__main__":
    main()
