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


# Constants for conflict detection
MIN_TOKENS_PER_REQUEST = 512  # Conservative estimate for resource constraint checks


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
            # Tokenizer Parameters
            ParameterDefinition(
                name="tokenizer_worker_num",
                values=[1, 2, 4, 8, 16],
                description="The worker num of the tokenizer manager"
            ),
            
            # Memory and Scheduling Parameters
            ParameterDefinition(
                name="chunked_prefill_size",
                values=[None, 512, 1024, 2048, 4096, 8192, 16384],
                description="Chunked prefill size for better scheduling"
            ),
            ParameterDefinition(
                name="max_prefill_tokens",
                values=[4096, 8192, 16384, 32768, 65536],
                description="Maximum tokens in a prefill batch"
            ),
            ParameterDefinition(
                name="schedule_policy",
                values=["fcfs", "lpm", "random", "dfs-weight", "lof"],
                description="The scheduling policy of the requests"
            ),
            
            # Cache and Memory Parameters
            ParameterDefinition(
                name="page_size",
                values=[None, 16, 32, 64, 128, 256],
                description="The number of tokens in a page"
            ),
            ParameterDefinition(
                name="swa_full_tokens_ratio",
                values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                description="The ratio of SWA layer KV tokens / full layer KV tokens"
            ),
            ParameterDefinition(
                name="radix_eviction_policy",
                values=["lru", "lfu"],
                description="The eviction policy of radix trees (lru: Least Recently Used, lfu: Least Frequently Used)",
                conflicts_with={("disable_radix_cache", True)}
            ),
            
            # Attention Backend Parameters
            ParameterDefinition(
                name="attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3", "fa4"],
                description="The main attention backend"
            ),
            ParameterDefinition(
                name="decode_attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3"],
                description="Attention backend for decode phase"
            ),
            ParameterDefinition(
                name="prefill_attention_backend",
                values=[None, "flashinfer", "triton", "torch_native", "fa3"],
                description="Attention backend for prefill phase"
            ),
            ParameterDefinition(
                name="sampling_backend",
                values=[None, "flashinfer", "pytorch"],
                description="Choose the kernels for sampling layers"
            ),
            ParameterDefinition(
                name="mm_attention_backend",
                values=[None, "sdpa", "fa3", "triton_attn"],
                description="Set multimodal attention backend"
            ),
            ParameterDefinition(
                name="nsa_prefill_backend",
                values=["flashmla_sparse", "flashmla_kv", "flashmla_auto", "fa3", "tilelang"],
                description="NSA prefill backend",
                conflicts_with={("attention_backend", "nsa")}  # Only used when attention_backend is nsa
            ),
            ParameterDefinition(
                name="nsa_decode_backend",
                values=["fa3", "flashmla_kv", "flashmla_sparse"],
                description="NSA decode backend",
                conflicts_with={("attention_backend", "nsa")}  # Only used when attention_backend is nsa
            ),
            
            # Hierarchical Cache Parameters
            ParameterDefinition(
                name="enable_hierarchical_cache",
                values=[False, True],
                description="Enable hierarchical cache for CPU-GPU KV cache transfer",
                conflicts_with={("enable_lmcache", True)}
            ),
            ParameterDefinition(
                name="hicache_ratio",
                values=[1.0, 1.5, 2.0, 3.0, 4.0],
                description="The ratio of the size of host KV cache memory pool to device pool"
            ),
            ParameterDefinition(
                name="hicache_size",
                values=[0, 8, 16, 32, 64],
                description="The size of host KV cache memory pool in GB (0 means use hicache_ratio)"
            ),
            ParameterDefinition(
                name="hicache_write_policy",
                values=["write_back", "write_through", "write_through_selective"],
                description="The write policy of hierarchical cache"
            ),
            ParameterDefinition(
                name="hicache_io_backend",
                values=["kernel", "direct"],
                description="The IO backend for KV cache transfer between CPU and GPU"
            ),
            ParameterDefinition(
                name="hicache_mem_layout",
                values=["layer_first", "page_first", "page_first_direct", "page_head"],
                description="The layout of host memory pool for hierarchical cache"
            ),
            ParameterDefinition(
                name="hicache_storage_backend",
                values=[None, "file", "mooncake"],
                description="The storage backend for hierarchical KV cache"
            ),
            ParameterDefinition(
                name="hicache_storage_prefetch_policy",
                values=["best_effort", "wait_complete", "timeout"],
                description="Control when prefetching from the storage backend should stop"
            ),
            
            # LMCache
            ParameterDefinition(
                name="enable_lmcache",
                values=[False, True],
                description="Using LMCache as an alternative hierarchical cache solution",
                conflicts_with={("enable_hierarchical_cache", True)}
            ),
            
            # Radix Cache
            ParameterDefinition(
                name="disable_radix_cache",
                values=[False, True],
                description="Disable radix cache",
                conflicts_with={("enable_hierarchical_cache", True)}
            ),
            
            # CUDA Graph Parameters
            ParameterDefinition(
                name="cuda_graph_max_bs",
                values=[None, 8, 16, 24, 32, 48, 64, 80, 96],
                description="Maximum batch size for CUDA graph capture"
            ),
            
            # Overlap and Optimization Parameters
            ParameterDefinition(
                name="disable_overlap_schedule",
                values=[False, True],
                description="Disable overlap scheduling between prefill and decode"
            ),
            ParameterDefinition(
                name="enable_mixed_chunk",
                values=[False, True],
                description="Enable mixed chunk mode for better scheduling"
            ),
            ParameterDefinition(
                name="enable_two_batch_overlap",
                values=[False, True],
                description="Enable two batch overlap for better throughput",
                conflicts_with={("disable_overlap_schedule", True)}
            ),
            ParameterDefinition(
                name="enable_single_batch_overlap",
                values=[False, True],
                description="Enable single batch overlap",
                conflicts_with={("disable_overlap_schedule", True)}
            ),
            
            # Torch Compile and CUDA Graph
            ParameterDefinition(
                name="enable_torch_compile",
                values=[False, True],
                description="Enable torch.compile for model optimization"
            ),
            ParameterDefinition(
                name="enable_piecewise_cuda_graph",
                values=[False, True],
                description="Enable piecewise CUDA graph for better memory usage"
            ),
            ParameterDefinition(
                name="torch_compile_max_bs",
                values=[16, 32, 48, 64, 96],
                description="Maximum batch size for torch compile"
            ),
            
            # Continuous Decode
            ParameterDefinition(
                name="num_continuous_decode_steps",
                values=[1, 2, 4, 8, 16],
                description="Number of continuous decode steps"
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
        
        # Rule 2: Hierarchical cache parameters only make sense when enable_hierarchical_cache is True
        enable_hicache = combination.get("enable_hierarchical_cache", False)
        if not enable_hicache:
            # Skip combinations with non-default hicache settings when hicache is disabled
            # We allow the default values to pass through for simplicity
            pass
        
        # Rule 3: NSA backends only matter when attention_backend is "nsa"
        attention_backend = combination.get("attention_backend")
        if attention_backend != "nsa":
            # NSA-specific backends are only used when attention_backend is "nsa"
            # Allow combinations but they won't be used in practice
            pass
        
        # Rule 4: Overlap features conflict with disable_overlap_schedule
        disable_overlap = combination.get("disable_overlap_schedule", False)
        if disable_overlap:
            if combination.get("enable_two_batch_overlap", False):
                return False
            if combination.get("enable_single_batch_overlap", False):
                return False
        
        # Rule 5: torch_compile_max_bs only matters when enable_torch_compile is True
        enable_torch_compile = combination.get("enable_torch_compile", False)
        if not enable_torch_compile:
            # torch_compile_max_bs has no effect when torch_compile is disabled
            pass
        
        # Rule 6: Radix cache eviction policy only matters when radix cache is enabled
        disable_radix = combination.get("disable_radix_cache", False)
        if disable_radix:
            # radix_eviction_policy has no effect when radix cache is disabled
            pass
        
        # Rule 7: enable_hierarchical_cache and disable_radix_cache are incompatible
        if enable_hicache and disable_radix:
            return False
        
        # Rule 8: enable_hierarchical_cache and enable_lmcache are mutually exclusive
        enable_lmcache = combination.get("enable_lmcache", False)
        if enable_hicache and enable_lmcache:
            return False
        
        # Rule 9: hicache_mem_layout "page_first_direct" requires hicache_io_backend "direct"
        hicache_mem_layout = combination.get("hicache_mem_layout")
        hicache_io_backend = combination.get("hicache_io_backend")
        if hicache_mem_layout == "page_first_direct" and hicache_io_backend != "direct":
            return False
        
        # Rule 10: Reasonable CUDA graph batch size constraints
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
