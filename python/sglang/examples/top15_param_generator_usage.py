#!/usr/bin/env python3
"""
Example usage of the top15_throughput_param_generator.py

This script demonstrates various ways to use the Top 15 parameter generator
for throughput optimization experiments.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from top15_throughput_param_generator import Top15ThroughputParamGenerator


def example_1_basic_generation():
    """Example 1: Generate a small set of configurations."""
    print("\n" + "="*80)
    print("Example 1: Generate 100 configurations")
    print("="*80)
    
    generator = Top15ThroughputParamGenerator()
    
    # Generate 100 valid combinations
    combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=100
    )
    
    print(f"Generated {len(combinations)} valid configurations")
    print("\nFirst configuration:")
    print(combinations[0])


def example_2_export_formats():
    """Example 2: Export configurations in different formats."""
    print("\n" + "="*80)
    print("Example 2: Export configurations to JSON and CSV")
    print("="*80)
    
    generator = Top15ThroughputParamGenerator()
    
    # Generate combinations
    combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=500
    )
    
    # Export to JSON
    json_path = "/tmp/top15_configs.json"
    generator.export_to_json(json_path, combinations)
    print(f"✓ Exported {len(combinations)} configurations to {json_path}")
    
    # Export to CSV
    csv_path = "/tmp/top15_configs.csv"
    generator.export_to_csv(csv_path, combinations)
    print(f"✓ Exported {len(combinations)} configurations to {csv_path}")


def example_3_parameter_info():
    """Example 3: Get information about parameters."""
    print("\n" + "="*80)
    print("Example 3: Get parameter information")
    print("="*80)
    
    generator = Top15ThroughputParamGenerator()
    
    # Get parameter info
    info = generator.get_parameter_info()
    
    print(f"\nTotal parameters: {len(info)}")
    print("\nParameter summary:")
    for param_name, param_info in info.items():
        print(f"  • {param_name}: {param_info['num_values']} values")
        print(f"    {param_info['description']}")


def example_4_custom_filtering():
    """Example 4: Apply custom filtering logic."""
    print("\n" + "="*80)
    print("Example 4: Custom filtering - only flashinfer backend")
    print("="*80)
    
    generator = Top15ThroughputParamGenerator()
    
    # Generate all combinations with conflict filtering
    all_combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=10000
    )
    
    # Apply custom filter: only keep flashinfer attention backend
    flashinfer_only = [
        combo for combo in all_combinations
        if combo.get('attention_backend') == 'flashinfer'
    ]
    
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Flashinfer-only combinations: {len(flashinfer_only)}")
    
    if flashinfer_only:
        print("\nExample flashinfer configuration:")
        print(flashinfer_only[0])


def example_5_progressive_optimization():
    """Example 5: Progressive optimization strategy."""
    print("\n" + "="*80)
    print("Example 5: Progressive optimization - start with critical params")
    print("="*80)
    
    generator = Top15ThroughputParamGenerator()
    
    # Strategy: Start with most critical parameters fixed
    # Fix the top 3 most critical parameters to known good values
    critical_settings = {
        'attention_backend': 'flashinfer',
        'chunked_prefill_size': 4096,
        'max_prefill_tokens': 16384
    }
    
    # Generate combinations
    all_combinations = generator.generate_combinations(
        filter_conflicts=True,
        max_combinations=1000
    )
    
    # Filter for critical settings
    optimized_set = [
        combo for combo in all_combinations
        if all(combo.get(k) == v for k, v in critical_settings.items())
    ]
    
    print(f"Total combinations: {len(all_combinations)}")
    print(f"With critical settings fixed: {len(optimized_set)}")
    print(f"\nCritical settings: {critical_settings}")
    print(f"\nThis reduces the search space from {len(all_combinations)} to {len(optimized_set)}")
    print(f"configurations while focusing on the most impactful parameters.")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("SGLang Top 15 Throughput Parameter Generator - Usage Examples")
    print("="*80)
    
    try:
        example_1_basic_generation()
        example_2_export_formats()
        example_3_parameter_info()
        example_4_custom_filtering()
        example_5_progressive_optimization()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
