# ConfigFileGenerator Documentation

## Overview

The `ConfigFileGenerator` class extends the functionality of `Top15ThroughputParamGenerator` by allowing parameter combinations to be saved as individual JSON config files. This is useful for running experiments with different configurations or setting up SGLang server instances with optimized parameters.

## Features

- Save parameter combinations as individual JSON config files
- Customizable config file format with template support
- Configurable filename patterns
- CLI support for easy batch generation
- Merges generated parameters with default or custom config templates

## Config File Format

Each generated config file follows this format:

```json
{
  "context_length": 4096,
  "trust_remote_code": true,
  "dtype": "auto",
  "kv_cache_dtype": "auto",
  "quantization": null,
  "quantization_param_path": null,
  "mem_fraction_static": 0.8,
  "enable_chunked_prefill": null,
  "max_running_requests": 256,
  "chunked_prefill_size": 4096,
  "random_seed": 42,
  "enable_ep_moe": null,
  "disable_chunked_prefix_cache": false,
  "tp_size": 2,
  "attention_backend": "flashinfer",
  ...
}
```

The config includes:
- **Default template fields**: Standard SGLang configuration parameters
- **Generated parameters**: All 18 throughput-impacting parameters from combinations

## Usage

### 1. Command Line Interface

#### Basic Usage - Save as Individual Config Files

```bash
# Generate and save 10 config files
python -m sglang.top15_throughput_param_generator \
  --save-configs \
  --max-combinations 10

# This creates config_0.json, config_1.json, ..., config_9.json in the 'configs' directory
```

#### Customize Output Directory and Filename Pattern

```bash
# Custom directory and filename pattern
python -m sglang.top15_throughput_param_generator \
  --save-configs \
  --config-dir my_experiments \
  --config-pattern "exp_{index}.json" \
  --max-combinations 5
```

#### Generate All Valid Combinations

```bash
# Generate all valid combinations (may create many files!)
python -m sglang.top15_throughput_param_generator \
  --save-configs \
  --config-dir all_configs
```

### 2. Python API

#### Basic Usage

```python
from sglang.top15_throughput_param_generator import (
    ConfigFileGenerator,
    Top15ThroughputParamGenerator
)

# Create generator
config_gen = ConfigFileGenerator()

# Save a single config file
params = {
    "tp_size": 2,
    "attention_backend": "flashinfer",
    "chunked_prefill_size": 4096,
}
config_gen.save_config_file(params, "my_config.json")
```

#### Save Multiple Configurations

```python
# Generate parameter combinations
generator = Top15ThroughputParamGenerator()
combinations = generator.generate_combinations(
    filter_conflicts=True,
    max_combinations=10
)

# Save as individual config files
config_gen = ConfigFileGenerator(generator)
created_files = config_gen.save_multiple_configs(
    combinations,
    output_dir="configs",
    filename_pattern="config_{index}.json"
)

print(f"Created {len(created_files)} config files")
```

#### Generate and Save in One Step

```python
# Most convenient approach
config_gen = ConfigFileGenerator()
created_files = config_gen.generate_and_save_configs(
    output_dir="experiments",
    filename_pattern="exp_{index}.json",
    filter_conflicts=True,
    max_combinations=20
)
```

#### Custom Config Template

```python
# Define custom template with your own defaults
custom_template = {
    "context_length": 8192,
    "dtype": "float16",
    "mem_fraction_static": 0.9,
    "custom_field": "my_value",
}

# Use custom template
config_gen = ConfigFileGenerator()
config_gen.save_config_file(
    params={"tp_size": 4, "attention_backend": "flashinfer"},
    filepath="custom_config.json",
    template=custom_template
)
```

## API Reference

### Class: ConfigFileGenerator

#### `__init__(generator=None)`
Initialize the config file generator.

**Parameters:**
- `generator` (Optional[Top15ThroughputParamGenerator]): Parameter generator instance. Creates new one if not provided.

#### `save_config_file(params, filepath, template=None, indent=2)`
Save a single parameter combination as a JSON config file.

**Parameters:**
- `params` (Dict[str, Any]): Parameter combination dictionary
- `filepath` (str): Output file path
- `template` (Optional[Dict[str, Any]]): Custom config template (uses default if None)
- `indent` (int): JSON indentation level (default: 2)

#### `save_multiple_configs(combinations, output_dir, filename_pattern="config_{index}.json", template=None, indent=2)`
Save multiple parameter combinations as individual config files.

**Parameters:**
- `combinations` (List[Dict[str, Any]]): List of parameter combinations
- `output_dir` (str): Output directory for config files
- `filename_pattern` (str): Filename pattern with {index} placeholder
- `template` (Optional[Dict[str, Any]]): Custom config template
- `indent` (int): JSON indentation level

**Returns:**
- List[str]: List of created file paths

#### `generate_and_save_configs(output_dir, filename_pattern="config_{index}.json", filter_conflicts=True, max_combinations=None, template=None, indent=2)`
Generate parameter combinations and save them as config files in one step.

**Parameters:**
- `output_dir` (str): Output directory for config files
- `filename_pattern` (str): Filename pattern with {index} placeholder
- `filter_conflicts` (bool): Filter out conflicting combinations (default: True)
- `max_combinations` (Optional[int]): Maximum number of combinations to generate
- `template` (Optional[Dict[str, Any]]): Custom config template
- `indent` (int): JSON indentation level

**Returns:**
- List[str]: List of created file paths

## Examples

See `python/sglang/examples/config_file_generator_usage.py` for comprehensive examples including:
- Basic single file saving
- Multiple file generation
- Custom templates
- Custom filename patterns
- Integration with Top15ThroughputParamGenerator

Run examples:
```bash
python python/sglang/examples/config_file_generator_usage.py
```

## CLI Options

```
--save-configs              Enable config file generation mode
--config-dir DIR            Output directory (default: configs)
--config-pattern PATTERN    Filename pattern (default: config_{index}.json)
--max-combinations N        Maximum combinations to generate
--no-filter                 Disable conflict filtering
```

## Default Config Template

The default template includes these fields:
- `context_length`: 4096
- `trust_remote_code`: true
- `dtype`: "auto"
- `kv_cache_dtype`: "auto"
- `quantization`: null
- `quantization_param_path`: null
- `mem_fraction_static`: 0.8
- `enable_chunked_prefill`: null
- `max_running_requests`: 256
- `chunked_prefill_size`: 4096
- `random_seed`: 42
- `enable_ep_moe`: null
- `disable_chunked_prefix_cache`: false

Plus all 18 throughput-impacting parameters from the generator.

## Testing

Run tests:
```bash
python test/srt/test_config_file_generator.py
```

## See Also

- [TOP15_PARAMETERS.md](TOP15_PARAMETERS.md) - Documentation on the 18 key throughput parameters
- [top15_throughput_param_generator.py](python/sglang/top15_throughput_param_generator.py) - Main module
