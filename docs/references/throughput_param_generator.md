# SGLang Throughput Optimization Parameter Combination Generator

This tool generates all valid combinations of key performance parameters for SGLang throughput optimization testing. It automatically filters out logically conflicting parameter combinations.

## Overview

The parameter combination generator helps you systematically test different SGLang server configurations to find optimal throughput settings for your specific workload. It defines important performance parameters and their valid values, then generates all possible combinations while eliminating configurations that would be logically incompatible.

## Key Features

- **7 Performance-Critical Parameters** covering tokenizer, memory, scheduling, and caching
- **Automatic Conflict Detection** to filter out invalid parameter combinations
- **Multiple Export Formats** (JSON, CSV) for easy integration with automation tools
- **Extensible Design** allowing custom parameters and conflict rules
- **Command-Line Interface** for easy usage

## Supported Parameters

### Tokenizer
- `tokenizer_worker_num`: The worker num of the tokenizer manager
  - Values: `[1, 2, 4, 8]`

### Memory and Scheduling
- `chunked_prefill_size`: Chunked prefill size for better scheduling
  - Values: `[None, 512, 1024, 2048, 4096, 8192]`
- `max_prefill_tokens`: Maximum tokens in a prefill batch
  - Values: `[4096, 8192, 16384, 32768]`
- `schedule_policy`: The scheduling policy of the requests
  - Values: `["fcfs", "lpm", "random", "dfs-weight", "lof"]`

### Cache and Memory
- `page_size`: The number of tokens in a page
  - Values: `[None, 16, 32, 64, 128]`
- `swa_full_tokens_ratio`: The ratio of SWA layer KV tokens / full layer KV tokens
  - Values: `[0.6, 0.7, 0.8, 0.9, 1.0]`
- `radix_eviction_policy`: The eviction policy of radix trees (lru: Least Recently Used, lfu: Least Frequently Used)
  - Values: `["lru", "lfu"]`

## Conflict Detection Rules

The generator automatically filters out invalid combinations based on these rules:

1. **Size Constraints**: `chunked_prefill_size` must be ≤ `max_prefill_tokens` when both are set

## Usage

### As a Standalone Script

```bash
# Show parameter information
python -m sglang.throughput_param_generator --show-info

# Generate all valid combinations (with filtering)
python -m sglang.throughput_param_generator

# Generate first 100 combinations
python -m sglang.throughput_param_generator --max-combinations 100

# Export to JSON file
python -m sglang.throughput_param_generator --output configs.json --format json

# Export to CSV file
python -m sglang.throughput_param_generator --output configs.csv --format csv

# Generate without filtering (include conflicting combinations)
python -m sglang.throughput_param_generator --no-filter
```

### As a CLI Command (if sglang is installed)

```bash
# Show help
sglang gen-throughput-params --help

# Generate combinations
sglang gen-throughput-params --output my_configs.json

# Generate limited number of combinations
sglang gen-throughput-params --max-combinations 50 --output test_configs.json
```

### As a Python Module

```python
from sglang.throughput_param_generator import (
    ThroughputParamGenerator,
    ParameterDefinition
)

# Create generator with default parameters
generator = ThroughputParamGenerator()

# Generate all valid combinations
combinations = generator.generate_combinations(
    filter_conflicts=True,
    max_combinations=100
)

# Print summary
generator.print_summary(combinations)

# Export to file
generator.export_to_json("configs.json", combinations)
generator.export_to_csv("configs.csv", combinations)

# Get parameter information
info = generator.get_parameter_info()
print(info)

# Add custom parameter
custom_param = ParameterDefinition(
    name="my_param",
    values=[1, 2, 3],
    description="My custom parameter"
)
generator.add_parameter(custom_param)

# Remove a parameter
generator.remove_parameter("attention_backend")

# Generate with custom parameters
new_combinations = generator.generate_combinations()
```

## Output Format

### JSON Format

```json
[
  {
    "tokenizer_worker_num": 2,
    "chunked_prefill_size": 2048,
    "max_prefill_tokens": 8192,
    "schedule_policy": "fcfs",
    "page_size": 64,
    "swa_full_tokens_ratio": 0.8,
    "radix_eviction_policy": "lru"
  },
  ...
]
```

### CSV Format

```csv
tokenizer_worker_num,chunked_prefill_size,max_prefill_tokens,schedule_policy,page_size,swa_full_tokens_ratio,radix_eviction_policy
2,2048,8192,fcfs,64,0.8,lru
...
```

## Integration with Throughput Benchmarks

You can use the generated configurations with SGLang's throughput benchmarking tools:

```bash
# Generate parameter combinations
python -m sglang.throughput_param_generator --output configs.json --max-combinations 50

# Use with bench_offline_throughput.py
for config in $(jq -c '.[]' configs.json); do
    tokenizer_worker_num=$(echo $config | jq -r '.tokenizer_worker_num')
    chunked_prefill_size=$(echo $config | jq -r '.chunked_prefill_size')
    max_prefill_tokens=$(echo $config | jq -r '.max_prefill_tokens')
    schedule_policy=$(echo $config | jq -r '.schedule_policy')
    page_size=$(echo $config | jq -r '.page_size')
    
    # Build command with parameters
    python -m sglang.bench_offline_throughput \
        --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --tokenizer-worker-num $tokenizer_worker_num \
        --chunked-prefill-size $chunked_prefill_size \
        --max-prefill-tokens $max_prefill_tokens \
        --schedule-policy $schedule_policy \
        --page-size $page_size \
        --result-filename results.jsonl
done
```

## Extending the Generator

You can extend the generator with your own parameters and conflict rules:

```python
from sglang.throughput_param_generator import ThroughputParamGenerator

class CustomThroughputGenerator(ThroughputParamGenerator):
    def _define_default_parameters(self):
        # Get base parameters
        params = super()._define_default_parameters()
        
        # Add custom parameters
        from sglang.throughput_param_generator import ParameterDefinition
        params.append(ParameterDefinition(
            name="my_custom_param",
            values=[1, 2, 3, 4],
            description="My custom optimization parameter"
        ))
        
        return params
    
    def _is_valid_combination(self, combination):
        # Check base validation
        if not super()._is_valid_combination(combination):
            return False
        
        # Add custom validation rules
        if combination.get("my_custom_param") == 4:
            if combination.get("tp_size") > 2:
                return False
        
        return True

# Use custom generator
custom_gen = CustomThroughputGenerator()
combinations = custom_gen.generate_combinations()
```

## Performance Considerations

The number of theoretical combinations grows exponentially with the number of parameters and their values. The default configuration generates up to 24,000 theoretical combinations, which are filtered down to valid ones.

For large-scale parameter searches, consider:
- Using `--max-combinations` to limit the search space
- Focusing on specific parameter subsets by removing irrelevant parameters
- Using sampling strategies for initial exploration
- Running benchmarks in parallel on multiple machines

## Testing

The generator includes comprehensive unit tests:

```bash
# Run tests
python test/srt/test_throughput_param_generator.py -v
```

All 17 unit tests cover:
- Parameter definition and validation
- Combination generation
- Conflict filtering
- Export functionality
- Custom parameter handling

## License

This tool is part of SGLang and follows the same license terms.
