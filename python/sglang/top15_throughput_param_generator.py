#!/usr/bin/env python3
"""
SGLang Top 15 Throughput Parameters - Optimization Parameter Combination Generator

This module focuses on the 15 most impactful parameters for throughput optimization.
These parameters were selected based on their direct impact on:
1. Compute performance (attention backends, torch compile)
2. Memory management (chunked prefill, page size, max prefill tokens)
3. Scheduling efficiency (schedule policy, overlap features)
4. Parallel processing (tokenizer workers, continuous decode steps)

Usage:
    python -m sglang.top15_throughput_param_generator --output configs.json
    python -m sglang.top15_throughput_param_generator --format csv --output configs.csv
"""

import argparse
import csv
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


class Top15ThroughputParamGenerator:
    """
    Generator for top 15 most impactful throughput optimization parameters.
    
    This class focuses on the parameters with the highest impact on throughput
    performance, generating all valid combinations while filtering out conflicts.
    
    Selected Parameters (in order of impact):
    1. attention_backend - Main attention computation backend
    2. chunked_prefill_size - Prefill chunking for better scheduling
    3. max_prefill_tokens - Maximum tokens in prefill batch
    4. schedule_policy - Request scheduling algorithm
    5. decode_attention_backend - Decode phase attention backend
    6. prefill_attention_backend - Prefill phase attention backend
    7. page_size - Memory page size for KV cache
    8. cuda_graph_max_bs - CUDA graph optimization batch size
    9. enable_mixed_chunk - Mixed chunking mode
    10. disable_overlap_schedule - Overlap scheduling control
    11. enable_torch_compile - Torch compilation optimization
    12. num_continuous_decode_steps - Continuous decode efficiency
    13. enable_two_batch_overlap - Two-batch overlap optimization
    14. tokenizer_worker_num - Tokenizer parallelization
    15. sampling_backend - Sampling computation backend
    """
    
    def __init__(self):
        """Initialize the parameter generator with top 15 parameters."""
        self.parameters = self._define_top15_parameters()
    
    def _define_top15_parameters(self) -> List[ParameterDefinition]:
        """
        Define the top 15 most impactful performance parameters.
        
        These parameters were selected based on their direct impact on:
        - Compute performance (attention backends, compilation)
        - Memory efficiency (chunking, page size)
        - Scheduling optimization (policies, overlap features)
        - Parallelization (tokenizer workers, continuous decode)
        
        Returns:
            List of ParameterDefinition objects for top 15 parameters.
        """
        parameters = [
            # Top 1: Attention Backend - Most critical for compute performance
            ParameterDefinition(
                name="attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3", "fa4"],
                description="The main attention backend (highest impact on compute performance)"
            ),
            
            # Top 2: Chunked Prefill Size - Critical for memory and scheduling
            ParameterDefinition(
                name="chunked_prefill_size",
                values=[None, 512, 1024, 2048, 4096, 8192, 16384],
                description="Chunked prefill size for better scheduling and memory management"
            ),
            
            # Top 3: Max Prefill Tokens - Directly impacts batch processing
            ParameterDefinition(
                name="max_prefill_tokens",
                values=[4096, 8192, 16384, 32768, 65536],
                description="Maximum tokens in a prefill batch (affects throughput directly)"
            ),
            
            # Top 4: Schedule Policy - Core scheduling algorithm
            ParameterDefinition(
                name="schedule_policy",
                values=["fcfs", "lpm", "random", "dfs-weight", "lof"],
                description="The scheduling policy of requests (impacts request ordering)"
            ),
            
            # Top 5: Decode Attention Backend - Critical for decode phase
            ParameterDefinition(
                name="decode_attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3"],
                description="Attention backend for decode phase (high impact on decode perf)"
            ),
            
            # Top 6: Prefill Attention Backend - Critical for prefill phase
            ParameterDefinition(
                name="prefill_attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3"],
                description="Attention backend for prefill phase (high impact on prefill perf)"
            ),
            
            # Top 7: Page Size - Memory management efficiency
            ParameterDefinition(
                name="page_size",
                values=[None, 16, 32, 64, 128, 256],
                description="The number of tokens in a page (affects memory efficiency)"
            ),
            
            # Top 8: CUDA Graph Max Batch Size - CUDA optimization
            ParameterDefinition(
                name="cuda_graph_max_bs",
                values=[None, 8, 16, 24, 32, 48, 64, 80, 96],
                description="Maximum batch size for CUDA graph capture (reduces overhead)"
            ),
            
            # Top 9: Enable Mixed Chunk - Scheduling optimization
            ParameterDefinition(
                name="enable_mixed_chunk",
                values=[False, True],
                description="Enable mixed chunk mode for better scheduling"
            ),
            
            # Top 10: Disable Overlap Schedule - Major scheduling feature
            ParameterDefinition(
                name="disable_overlap_schedule",
                values=[False, True],
                description="Disable overlap scheduling between prefill and decode"
            ),
            
            # Top 11: Enable Torch Compile - Compilation optimization
            ParameterDefinition(
                name="enable_torch_compile",
                values=[False, True],
                description="Enable torch.compile for model optimization"
            ),
            
            # Top 12: Continuous Decode Steps - Decode efficiency
            ParameterDefinition(
                name="num_continuous_decode_steps",
                values=[1, 2, 4, 8, 16],
                description="Number of continuous decode steps (affects decode throughput)"
            ),
            
            # Top 13: Enable Two Batch Overlap - Throughput optimization
            ParameterDefinition(
                name="enable_two_batch_overlap",
                values=[False, True],
                description="Enable two batch overlap for better throughput",
                conflicts_with={("disable_overlap_schedule", True)}
            ),
            
            # Top 14: Tokenizer Worker Number - Parallel processing
            ParameterDefinition(
                name="tokenizer_worker_num",
                values=[1, 2, 4, 8, 16],
                description="The worker num of the tokenizer manager (affects tokenization speed)"
            ),
            
            # Top 15: Sampling Backend - Sampling performance
            ParameterDefinition(
                name="sampling_backend",
                values=[None, "flashinfer", "pytorch"],
                description="Choose the kernels for sampling layers"
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
        
        # Additional logical conflict rules for top 15 parameters
        
        # Rule 1: Ensure chunked_prefill_size <= max_prefill_tokens if both are set
        chunked_prefill = combination.get("chunked_prefill_size")
        max_prefill = combination.get("max_prefill_tokens")
        if chunked_prefill is not None and max_prefill is not None:
            if chunked_prefill > max_prefill:
                return False
        
        # Rule 2: Overlap features conflict with disable_overlap_schedule
        disable_overlap = combination.get("disable_overlap_schedule", False)
        if disable_overlap:
            if combination.get("enable_two_batch_overlap", False):
                return False
        
        # Rule 3: Reasonable CUDA graph batch size constraints
        cuda_graph_max_bs = combination.get("cuda_graph_max_bs")
        if cuda_graph_max_bs is not None and chunked_prefill is not None:
            # Avoid extremely large cuda_graph_max_bs with small chunked_prefill_size
            # as it would waste memory
            if chunked_prefill <= 2048 and cuda_graph_max_bs > 32:
                return False
        
        return True
    
    def generate_combinations(
        self,
        filter_conflicts: bool = True,
        max_combinations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for top 15 parameters.
        
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
        print("\n" + "=" * 80)
        print("SGLang Top 15 Throughput Optimization Parameters - Combination Generator")
        print("=" * 80)
        print(f"\nTotal valid combinations: {len(combinations)}")
        print(f"Number of parameters: {len(self.parameters)}")
        
        print("\nTop 15 Parameters (ordered by performance impact):")
        for i, param in enumerate(self.parameters, 1):
            print(f"\n{i}. {param.name}:")
            print(f"   Values ({len(param.values)}): {param.values}")
            if param.description:
                print(f"   Description: {param.description}")
            if param.conflicts_with:
                print(f"   Conflicts with: {list(param.conflicts_with)}")
        
        # Calculate theoretical max combinations
        total_possible = 1
        for param in self.parameters:
            total_possible *= len(param.values)
        
        print(f"\n{'=' * 80}")
        print(f"Theoretical combinations (without filtering): {total_possible:,}")
        print(f"Valid combinations (after filtering): {len(combinations):,}")
        if total_possible > 0:
            filtered_percent = (total_possible - len(combinations)) / total_possible * 100
            print(f"Filtered out: {filtered_percent:.2f}%")
        print("=" * 80 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate parameter combinations for top 15 throughput-impacting parameters"
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
    generator = Top15ThroughputParamGenerator()
    
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
