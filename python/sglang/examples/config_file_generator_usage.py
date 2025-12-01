#!/usr/bin/env python3
"""
Example usage of the ConfigFileGenerator class.

This script demonstrates how to use ConfigFileGenerator to save
parameter combinations as individual config files.
"""

import os
import sys
from pathlib import Path

# Add the python directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, python_dir)

# Import directly from the module file to avoid dependency issues
import importlib.util
module_path = os.path.join(python_dir, 'sglang', 'top15_throughput_param_generator.py')
spec = importlib.util.spec_from_file_location(
    "top15_throughput_param_generator",
    module_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ConfigFileGenerator = module.ConfigFileGenerator
Top15ThroughputParamGenerator = module.Top15ThroughputParamGenerator


def example_1_basic_usage():
    """Example 1: Basic usage - save a single config file."""
    print("=" * 80)
    print("Example 1: Save a single config file")
    print("=" * 80)
    
    # Create generator and config file generator
    config_gen = ConfigFileGenerator()
    
    # Define a parameter combination
    params = {
        "tp_size": 2,
        "attention_backend": "flashinfer",
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 8192,
        "dp_size": 1,
        "schedule_policy": "fcfs",
    }
    
    # Save as config file
    output_file = "/tmp/my_config.json"
    config_gen.save_config_file(params, output_file)
    
    print(f"✓ Config file saved to: {output_file}")
    print("\nContent preview:")
    with open(output_file, 'r') as f:
        print(f.read())
    print()


def example_2_save_multiple():
    """Example 2: Save multiple config files from combinations."""
    print("=" * 80)
    print("Example 2: Save multiple config files")
    print("=" * 80)
    
    # Create generator
    generator = Top15ThroughputParamGenerator()
    config_gen = ConfigFileGenerator(generator)
    
    # Generate some combinations
    combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=5
    )
    
    print(f"Generated {len(combinations)} parameter combinations")
    
    # Save them as individual config files
    output_dir = "/tmp/configs"
    created_files = config_gen.save_multiple_configs(
        combinations,
        output_dir,
        filename_pattern="config_{index}.json"
    )
    
    print(f"✓ Created {len(created_files)} config files in {output_dir}")
    for filepath in created_files:
        print(f"  - {filepath}")
    print()


def example_3_generate_and_save():
    """Example 3: Generate and save in one step."""
    print("=" * 80)
    print("Example 3: Generate and save in one step")
    print("=" * 80)
    
    # Create config file generator
    config_gen = ConfigFileGenerator()
    
    # Generate and save configs in one call
    output_dir = "/tmp/auto_configs"
    created_files = config_gen.generate_and_save_configs(
        output_dir=output_dir,
        filename_pattern="auto_config_{index}.json",
        filter_conflicts=True,
        max_combinations=3
    )
    
    print(f"✓ Generated and saved {len(created_files)} config files")
    for filepath in created_files:
        print(f"  - {filepath}")
    print()


def example_4_custom_template():
    """Example 4: Use custom config template."""
    print("=" * 80)
    print("Example 4: Custom config template")
    print("=" * 80)
    
    # Create generator
    config_gen = ConfigFileGenerator()
    
    # Define custom template with different defaults
    custom_template = {
        "context_length": 8192,  # Different from default 4096
        "trust_remote_code": True,
        "dtype": "float16",  # Different from default "auto"
        "kv_cache_dtype": "float16",
        "quantization": None,
        "mem_fraction_static": 0.9,  # Different from default 0.8
        "max_running_requests": 512,  # Different from default 256
        "random_seed": 123,  # Different from default 42
        "custom_field": "my_custom_value",  # New custom field
    }
    
    # Define parameters
    params = {
        "tp_size": 4,
        "attention_backend": "flashinfer",
        "chunked_prefill_size": 8192,
    }
    
    # Save with custom template
    output_file = "/tmp/custom_config.json"
    config_gen.save_config_file(params, output_file, template=custom_template)
    
    print(f"✓ Config with custom template saved to: {output_file}")
    print("\nContent preview:")
    with open(output_file, 'r') as f:
        print(f.read())
    print()


def example_5_custom_filename_pattern():
    """Example 5: Custom filename patterns."""
    print("=" * 80)
    print("Example 5: Custom filename patterns")
    print("=" * 80)
    
    # Create generator
    generator = Top15ThroughputParamGenerator()
    config_gen = ConfigFileGenerator(generator)
    
    # Generate combinations
    combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=3
    )
    
    # Save with different filename patterns
    patterns = [
        "sglang_config_{index}.json",
        "experiment_{index}_config.json",
        "run_{index}.json"
    ]
    
    for pattern in patterns:
        output_dir = f"/tmp/configs_{pattern.split('_')[0]}"
        created_files = config_gen.save_multiple_configs(
            combinations,
            output_dir,
            filename_pattern=pattern
        )
        print(f"✓ Pattern '{pattern}':")
        print(f"  Created {len(created_files)} files in {output_dir}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("ConfigFileGenerator Usage Examples")
    print("=" * 80)
    print()
    
    try:
        example_1_basic_usage()
        example_2_save_multiple()
        example_3_generate_and_save()
        example_4_custom_template()
        example_5_custom_filename_pattern()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
