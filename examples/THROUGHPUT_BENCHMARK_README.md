# Throughput Parameter Benchmark Example

This example demonstrates how to use the SGLang throughput parameter combination generator for automated benchmarking.

## Quick Start

### 1. Generate parameter combinations (dry run)
```bash
python throughput_param_benchmark_example.py --max-configs 10
```

This will show what commands would be executed without actually running them.

### 2. Generate and save configurations
```bash
python throughput_param_benchmark_example.py --max-configs 20 --output-configs my_configs.json
```

### 3. Run actual benchmarks (requires model)
```bash
python throughput_param_benchmark_example.py \
    --max-configs 5 \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --run
```

**Warning**: Running actual benchmarks requires a GPU and will take significant time.

### 4. Analyze existing results
```bash
python throughput_param_benchmark_example.py \
    --analyze-only \
    --result-file /tmp/benchmark_results.jsonl
```

## What This Example Does

1. **Generates Parameter Combinations**: Uses the throughput parameter generator to create valid configurations
2. **Builds Benchmark Commands**: Constructs `sglang.bench_offline_throughput` commands with each configuration
3. **Runs Benchmarks** (optional): Executes benchmarks and collects results
4. **Analyzes Results**: Finds the best configurations by throughput

## Customization

You can customize the generator in the script:

```python
generator = ThroughputParamGenerator()

# Remove parameters you don't want to test
generator.remove_parameter("attention_backend")

# Add custom parameters
generator.add_parameter(ParameterDefinition(
    name="custom_param",
    values=[1, 2, 3],
    description="Custom parameter"
))

# Generate combinations
combinations = generator.generate_combinations(max_combinations=50)
```

## Output

The script produces:
- **Configuration file**: JSON file with all parameter combinations
- **Results file**: JSONL file with benchmark results (one JSON object per line)
- **Analysis report**: Summary of top performing configurations

## See Also

- [Parameter Generator Documentation](../docs/references/throughput_param_generator.md)
- [SGLang Benchmarking Tools](../python/sglang/bench_offline_throughput.py)
