#!/usr/bin/env python3
"""
SGLang Throughput Optimization Parameter Combination Generator

This module generates all valid combinations of key performance parameters
for throughput optimization testing. It automatically filters out logically
conflicting parameter combinations.

Usage:
    python -m sglang.throughput_param_generator --output configs.json
    python -m sglang.throughput_param_generator --format csv --output configs.csv
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ParameterDefinition:
    """Definition of a parameter with its valid values."""
    name: str
    values: List[Any]
    description: str = ""
    conflicts_with: Set[Tuple[str, Any]] = field(default_factory=set)
    
    def __post_init__(self):
        """Convert conflicts_with to set if passed as list."""
        if isinstance(self.conflicts_with, list):
            self.conflicts_with = set(self.conflicts_with)


class ThroughputParamGenerator:
    """
    Generator for throughput optimization parameter combinations.
    
    This class defines key SGLang performance parameters and generates
    all valid combinations while filtering out conflicting configurations.
    """
    
    def __init__(self):
        """Initialize the parameter generator with default parameter definitions."""
        self.parameters = self._define_default_parameters()
    
    def _define_default_parameters(self) -> List[ParameterDefinition]:
        """
        Define default performance-critical parameters for throughput optimization.
        
        Returns:
            List of ParameterDefinition objects with key performance parameters.
        """
        parameters = [
            # Memory and Scheduling Parameters
            ParameterDefinition(
                name="max_running_requests",
                values=[None, 128, 256, 512, 1024],
                description="Maximum number of concurrent running requests"
            ),
            ParameterDefinition(
                name="max_total_tokens",
                values=[None, 4096, 8192, 16384, 32768],
                description="Maximum total tokens in the system"
            ),
            ParameterDefinition(
                name="chunked_prefill_size",
                values=[None, 512, 1024, 2048, 4096, 8192],
                description="Chunked prefill size for better scheduling"
            ),
            ParameterDefinition(
                name="max_prefill_tokens",
                values=[4096, 8192, 16384, 32768],
                description="Maximum tokens in a prefill batch"
            ),
            
            # Parallelism Parameters
            ParameterDefinition(
                name="tp_size",
                values=[1, 2, 4, 8],
                description="Tensor parallelism size"
            ),
            ParameterDefinition(
                name="dp_size",
                values=[1, 2, 4],
                description="Data parallelism size"
            ),
            
            # Cache Parameters
            ParameterDefinition(
                name="disable_radix_cache",
                values=[False, True],
                description="Whether to disable radix cache"
            ),
            
            # CUDA Graph Parameters
            ParameterDefinition(
                name="disable_cuda_graph",
                values=[False, True],
                description="Whether to disable CUDA graph optimization"
            ),
            ParameterDefinition(
                name="cuda_graph_max_bs",
                values=[None, 32, 64, 128, 256],
                description="Maximum batch size for CUDA graph",
                conflicts_with={("disable_cuda_graph", True)}
            ),
            
            # Attention Backend
            ParameterDefinition(
                name="attention_backend",
                values=[None, "flashinfer", "triton", "torch_native"],
                description="Attention computation backend"
            ),
        ]
        
        return parameters
    
    def _is_valid_combination(self, combination: Dict[str, Any]) -> bool:
        """
        Check if a parameter combination is valid (no conflicts).
        
        Args:
            combination: Dictionary mapping parameter names to values
            
        Returns:
            True if the combination is valid, False otherwise
        """
        # Check explicit conflicts defined in parameter definitions
        for param_def in self.parameters:
            if param_def.name in combination:
                value = combination[param_def.name]
                for conflict_name, conflict_value in param_def.conflicts_with:
                    if conflict_name in combination:
                        if combination[conflict_name] == conflict_value:
                            return False
        
        # Additional logical conflict rules
        
        # Rule 1: Ensure chunked_prefill_size <= max_prefill_tokens if both are set
        chunked_prefill = combination.get("chunked_prefill_size")
        max_prefill = combination.get("max_prefill_tokens")
        if chunked_prefill is not None and max_prefill is not None:
            if chunked_prefill > max_prefill:
                return False
        
        # Rule 2: Ensure max_running_requests * typical_tokens_per_request <= max_total_tokens
        # (conservative check: assume at least 512 tokens per request)
        max_running = combination.get("max_running_requests")
        max_total = combination.get("max_total_tokens")
        if max_running is not None and max_total is not None:
            min_tokens_needed = max_running * 512  # Conservative estimate
            if min_tokens_needed > max_total:
                return False
        
        return True
    
    def generate_combinations(
        self,
        filter_conflicts: bool = True,
        max_combinations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations.
        
        Args:
            filter_conflicts: If True, filter out conflicting combinations
            max_combinations: Maximum number of combinations to generate (None = all)
            
        Returns:
            List of parameter combination dictionaries
        """
        # Get all parameter names and their values
        param_names = [p.name for p in self.parameters]
        param_values = [p.values for p in self.parameters]
        
        # Generate all combinations using itertools.product
        all_combinations = []
        for combo_values in itertools.product(*param_values):
            combination = dict(zip(param_names, combo_values))
            
            # Filter conflicts if requested
            if filter_conflicts:
                if not self._is_valid_combination(combination):
                    continue
            
            all_combinations.append(combination)
            
            # Check if we've reached the maximum
            if max_combinations and len(all_combinations) >= max_combinations:
                break
        
        return all_combinations
    
    def add_parameter(self, param_def: ParameterDefinition):
        """
        Add a custom parameter to the generator.
        
        Args:
            param_def: ParameterDefinition to add
        """
        self.parameters.append(param_def)
    
    def remove_parameter(self, param_name: str):
        """
        Remove a parameter from the generator.
        
        Args:
            param_name: Name of the parameter to remove
        """
        self.parameters = [p for p in self.parameters if p.name != param_name]
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all defined parameters.
        
        Returns:
            Dictionary with parameter information
        """
        info = {}
        for param in self.parameters:
            info[param.name] = {
                "values": param.values,
                "description": param.description,
                "num_values": len(param.values),
                "conflicts_with": list(param.conflicts_with) if param.conflicts_with else []
            }
        return info
    
    def export_to_json(self, filepath: str, combinations: List[Dict[str, Any]]):
        """
        Export combinations to JSON file.
        
        Args:
            filepath: Output file path
            combinations: List of parameter combinations
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(combinations, f, indent=2, ensure_ascii=False)
    
    def export_to_csv(self, filepath: str, combinations: List[Dict[str, Any]]):
        """
        Export combinations to CSV file.
        
        Args:
            filepath: Output file path
            combinations: List of parameter combinations
        """
        import csv
        
        if not combinations:
            return
        
        # Check if first combination is not empty
        if not combinations[0]:
            return
        
        # Get all keys from the first combination
        fieldnames = list(combinations[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combinations)
    
    def print_summary(self, combinations: List[Dict[str, Any]]):
        """
        Print a summary of generated combinations.
        
        Args:
            combinations: List of parameter combinations
        """
        print("\n" + "=" * 70)
        print("SGLang Throughput Optimization Parameter Combinations")
        print("=" * 70)
        print(f"\nTotal valid combinations: {len(combinations)}")
        print(f"Number of parameters: {len(self.parameters)}")
        
        print("\nParameters and their values:")
        for param in self.parameters:
            print(f"  - {param.name}: {len(param.values)} values")
            print(f"    {param.values}")
            if param.description:
                print(f"    Description: {param.description}")
        
        # Calculate theoretical max combinations
        total_possible = 1
        for param in self.parameters:
            total_possible *= len(param.values)
        
        print(f"\nTheoretical combinations (without filtering): {total_possible}")
        print(f"Valid combinations (after filtering): {len(combinations)}")
        if total_possible > 0:
            filtered_percent = (total_possible - len(combinations)) / total_possible * 100
            print(f"Filtered out: {filtered_percent:.2f}%")
        
        print("=" * 70 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate SGLang throughput optimization parameter combinations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Maximum number of combinations to generate (default: all)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable conflict filtering"
    )
    parser.add_argument(
        "--show-info",
        action="store_true",
        help="Show parameter information and exit"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ThroughputParamGenerator()
    
    # Show parameter info if requested
    if args.show_info:
        info = generator.get_parameter_info()
        print(json.dumps(info, indent=2))
        return 0
    
    # Generate combinations
    combinations = generator.generate_combinations(
        filter_conflicts=not args.no_filter,
        max_combinations=args.max_combinations
    )
    
    # Print summary
    generator.print_summary(combinations)
    
    # Export or print results
    if args.output:
        if args.format == "json":
            generator.export_to_json(args.output, combinations)
            print(f"Exported {len(combinations)} combinations to {args.output}")
        elif args.format == "csv":
            generator.export_to_csv(args.output, combinations)
            print(f"Exported {len(combinations)} combinations to {args.output}")
    else:
        # Print first 5 combinations as examples
        if combinations:
            print("Example combinations (first 5):")
            for i, combo in enumerate(combinations[:5], 1):
                print(f"\nCombination {i}:")
                print(json.dumps(combo, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
