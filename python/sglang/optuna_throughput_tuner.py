#!/usr/bin/env python3
"""
SGLang Optuna Throughput Parameter Tuner

This module uses Optuna to perform multi-objective hyperparameter optimization
for SGLang throughput parameters. It optimizes two objectives:
1. Minimize TTFT (Time To First Token)
2. Maximize TPS (Tokens Per Second)

The parameters are based on the top 18 most impactful throughput optimization 
parameters defined in top15_throughput_param_generator.py (note: despite the 
filename "top15", it actually defines 18 parameters).

Usage:
    python -m sglang.optuna_throughput_tuner --n-trials 100 --device-num 8
    python -m sglang.optuna_throughput_tuner --n-trials 50 --storage sqlite:///optuna_study.db
"""

import argparse
import json
import sys
from typing import Any, Dict, Tuple

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("Error: optuna is not installed. Please install it with: pip install optuna")
    sys.exit(1)


def run_test(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Placeholder function for running tests with the given parameters.
    
    This function should be replaced with actual benchmarking code that:
    1. Starts an SGLang server with the specified parameters
    2. Runs throughput benchmarks
    3. Measures TTFT and TPS metrics
    4. Returns the results
    
    Args:
        params: Dictionary of SGLang parameters to test
        
    Returns:
        Tuple of (ttft, tps) where:
        - ttft: Time To First Token in seconds (lower is better)
        - tps: Tokens Per Second (higher is better)
    
    Example implementation:
        ```python
        def run_test(params: Dict[str, Any]) -> Tuple[float, float]:
            # Start server with params
            server = start_sglang_server(**params)
            
            # Run benchmark
            results = run_benchmark(server)
            
            # Extract metrics
            ttft = results['ttft']
            tps = results['tps']
            
            # Cleanup
            server.stop()
            
            return ttft, tps
        ```
    """
    # Placeholder implementation returning dummy values
    # In real usage, this should run actual benchmarks
    import random
    
    # Simulate TTFT (0.01 to 0.5 seconds)
    ttft = random.uniform(0.01, 0.5)
    
    # Simulate TPS (100 to 10000 tokens/sec)
    tps = random.uniform(100, 10000)
    
    return ttft, tps


class OptunaThroughputTuner:
    """
    Optuna-based tuner for SGLang throughput parameters.
    
    This class uses Optuna's multi-objective optimization to find parameter
    configurations that minimize TTFT and maximize TPS simultaneously.
    """
    
    def __init__(self, device_num: int = 8):
        """
        Initialize the Optuna tuner.
        
        Args:
            device_num: Number of GPU devices available (default: 8).
                       This constrains the valid combinations of tp_size, pp_size, and dp_size.
        """
        self.device_num = device_num
    
    def _suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial using Optuna.
        
        This method defines the search space for all 18 top throughput parameters
        from top15_throughput_param_generator.py.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        # Top 1: Tensor Parallelism Size
        params['tp_size'] = trial.suggest_categorical('tp_size', [1, 2, 4, 8])
        
        # Top 2: Attention Backend
        params['attention_backend'] = trial.suggest_categorical(
            'attention_backend',
            [None, 'flashinfer', 'triton', 'torch_native', 'fa3', 'fa4']
        )
        
        # Top 3: Chunked Prefill Size
        params['chunked_prefill_size'] = trial.suggest_categorical(
            'chunked_prefill_size',
            [None, 512, 1024, 2048, 4096, 8192, 16384]
        )
        
        # Top 4: Max Prefill Tokens
        params['max_prefill_tokens'] = trial.suggest_categorical(
            'max_prefill_tokens',
            [4096, 8192, 16384, 32768, 65536]
        )
        
        # Top 5: Data Parallelism Size
        params['dp_size'] = trial.suggest_categorical('dp_size', [1, 2, 4, 8])
        
        # Top 6: Schedule Policy
        params['schedule_policy'] = trial.suggest_categorical(
            'schedule_policy',
            ['fcfs', 'lpm', 'random', 'dfs-weight', 'lof']
        )
        
        # Top 7: Pipeline Parallelism Size
        params['pp_size'] = trial.suggest_categorical('pp_size', [1, 2, 4, 8])
        
        # Top 8: Decode Attention Backend
        params['decode_attention_backend'] = trial.suggest_categorical(
            'decode_attention_backend',
            [None, 'flashinfer', 'triton', 'torch_native', 'fa3']
        )
        
        # Top 9: Prefill Attention Backend
        params['prefill_attention_backend'] = trial.suggest_categorical(
            'prefill_attention_backend',
            [None, 'flashinfer', 'triton', 'torch_native', 'fa3']
        )
        
        # Top 10: Page Size
        params['page_size'] = trial.suggest_categorical(
            'page_size',
            [None, 16, 32, 64, 128, 256]
        )
        
        # Top 11: CUDA Graph Max Batch Size
        params['cuda_graph_max_bs'] = trial.suggest_categorical(
            'cuda_graph_max_bs',
            [None, 8, 16, 24, 32, 48, 64, 80, 96]
        )
        
        # Top 12: Enable Mixed Chunk
        params['enable_mixed_chunk'] = trial.suggest_categorical(
            'enable_mixed_chunk',
            [False, True]
        )
        
        # Top 13: Disable Overlap Schedule
        params['disable_overlap_schedule'] = trial.suggest_categorical(
            'disable_overlap_schedule',
            [False, True]
        )
        
        # Top 14: Enable Torch Compile
        params['enable_torch_compile'] = trial.suggest_categorical(
            'enable_torch_compile',
            [False, True]
        )
        
        # Top 15: Continuous Decode Steps
        params['num_continuous_decode_steps'] = trial.suggest_categorical(
            'num_continuous_decode_steps',
            [1, 2, 4, 8, 16]
        )
        
        # Top 16: Enable Two Batch Overlap
        params['enable_two_batch_overlap'] = trial.suggest_categorical(
            'enable_two_batch_overlap',
            [False, True]
        )
        
        # Top 17: Tokenizer Worker Number
        params['tokenizer_worker_num'] = trial.suggest_categorical(
            'tokenizer_worker_num',
            [1, 2, 4, 8, 16]
        )
        
        # Top 18: Sampling Backend
        params['sampling_backend'] = trial.suggest_categorical(
            'sampling_backend',
            [None, 'flashinfer', 'pytorch']
        )
        
        return params
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter combination for conflicts.
        
        This implements the same validation logic as in top15_throughput_param_generator.py
        to ensure only valid parameter combinations are tested.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Rule 1: Parallelism size constraints
        tp_size = params.get('tp_size', 1)
        pp_size = params.get('pp_size', 1)
        dp_size = params.get('dp_size', 1)
        
        # The product should not exceed the available device_num
        if tp_size * pp_size * dp_size > self.device_num:
            return False
        
        # Rule 2: pp_size > 1 requires compatible configuration
        if pp_size > 4 and dp_size > 4:
            return False
        
        # Rule 3: Ensure chunked_prefill_size <= max_prefill_tokens if both are set
        chunked_prefill = params.get('chunked_prefill_size')
        max_prefill = params.get('max_prefill_tokens')
        if chunked_prefill is not None and max_prefill is not None:
            if chunked_prefill > max_prefill:
                return False
        
        # Rule 4: Adjust chunked_prefill_size constraint based on dp_size
        if dp_size > 1 and chunked_prefill is not None and max_prefill is not None:
            effective_chunked_prefill = chunked_prefill // dp_size
            if effective_chunked_prefill < 256:
                return False
        
        # Rule 5: Overlap features conflict with disable_overlap_schedule
        disable_overlap = params.get('disable_overlap_schedule', False)
        if disable_overlap:
            if params.get('enable_two_batch_overlap', False):
                return False
        
        # Rule 6: Reasonable CUDA graph batch size constraints
        cuda_graph_max_bs = params.get('cuda_graph_max_bs')
        if cuda_graph_max_bs is not None and chunked_prefill is not None:
            if chunked_prefill <= 2048 and cuda_graph_max_bs > 32:
                return False
        
        # Rule 7: CUDA graph batch size interaction with dp_size
        if cuda_graph_max_bs is not None and dp_size > 1:
            if dp_size >= 4 and cuda_graph_max_bs > 64:
                return False
        
        return True
    
    def objective(self, trial: Trial) -> Tuple[float, float]:
        """
        Optuna objective function for multi-objective optimization.
        
        This function:
        1. Suggests parameters using Optuna
        2. Validates the parameter combination
        3. Runs tests with the parameters
        4. Returns objectives: (ttft, -tps) for minimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Tuple of (ttft, -tps) where both are to be minimized.
            Note: TPS is negated because Optuna minimizes by default,
                  and we want to maximize TPS.
        """
        # Suggest parameters
        params = self._suggest_parameters(trial)
        
        # Validate parameters
        if not self._validate_parameters(params):
            # Prune invalid trials
            raise optuna.TrialPruned("Invalid parameter combination")
        
        # Run test with the parameters
        try:
            ttft, tps = run_test(params)
        except Exception as e:
            print(f"Error running test: {e}")
            raise optuna.TrialPruned(f"Test execution failed: {e}")
        
        # Return objectives (both to minimize)
        # Note: We negate TPS because we want to maximize it
        return ttft, -tps
    
    def optimize(
        self,
        n_trials: int = 100,
        study_name: str = "sglang_throughput_optimization",
        storage: str = None,
        load_if_exists: bool = True
    ) -> optuna.Study:
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of optimization trials to run
            study_name: Name of the Optuna study
            storage: Database URL for persistent storage (e.g., "sqlite:///optuna.db")
            load_if_exists: Whether to load existing study if it exists
            
        Returns:
            Optuna study object with optimization results
        """
        # Create study with multi-objective optimization
        # Direction: minimize both objectives
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=['minimize', 'minimize'],  # minimize TTFT, minimize (-TPS)
            load_if_exists=load_if_exists,
            sampler=optuna.samplers.NSGAIISampler(),  # Good for multi-objective
            pruner=optuna.pruners.MedianPruner()
        )
        
        print(f"\n{'=' * 80}")
        print(f"Starting Optuna Multi-Objective Optimization")
        print(f"{'=' * 80}")
        print(f"Study name: {study_name}")
        print(f"Number of trials: {n_trials}")
        print(f"Device count: {self.device_num}")
        print(f"Objectives: Minimize TTFT, Maximize TPS")
        print(f"{'=' * 80}\n")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        return study
    
    def print_results(self, study: optuna.Study):
        """
        Print optimization results in a human-readable format.
        
        Args:
            study: Completed Optuna study
        """
        print(f"\n{'=' * 80}")
        print("Optimization Results")
        print(f"{'=' * 80}")
        
        # Get Pareto front trials
        pareto_trials = study.best_trials
        
        print(f"\nNumber of finished trials: {len(study.trials)}")
        print(f"Number of Pareto optimal solutions: {len(pareto_trials)}")
        
        if pareto_trials:
            print("\nPareto Front (Best Trade-offs between TTFT and TPS):")
            print("-" * 80)
            
            for i, trial in enumerate(pareto_trials[:10], 1):  # Show top 10
                ttft = trial.values[0]
                neg_tps = trial.values[1]
                tps = -neg_tps  # Convert back to positive TPS
                
                print(f"\nSolution {i}:")
                print(f"  TTFT: {ttft:.4f} seconds")
                print(f"  TPS: {tps:.2f} tokens/sec")
                print(f"  Parameters:")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")
            
            if len(pareto_trials) > 10:
                print(f"\n... and {len(pareto_trials) - 10} more Pareto optimal solutions")
        
        print(f"\n{'=' * 80}\n")
    
    def export_results(self, study: optuna.Study, output_file: str):
        """
        Export optimization results to a JSON file.
        
        Args:
            study: Completed Optuna study
            output_file: Path to output JSON file
        """
        results = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'pareto_front': []
        }
        
        # Export Pareto optimal trials
        for trial in study.best_trials:
            trial_data = {
                'trial_number': trial.number,
                'ttft': trial.values[0],
                'tps': -trial.values[1],  # Convert back to positive
                'parameters': trial.params,
                'state': trial.state.name
            }
            results['pareto_front'].append(trial_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to: {output_file}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Optuna-based multi-objective optimization for SGLang throughput parameters"
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials to run (default: 100)'
    )
    parser.add_argument(
        '--device-num',
        type=int,
        default=8,
        help='Number of GPU devices available (default: 8)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='sglang_throughput_optimization',
        help='Name of the Optuna study (default: sglang_throughput_optimization)'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Database URL for persistent storage (e.g., sqlite:///optuna.db)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='optuna_results.json',
        help='Output file for results (default: optuna_results.json)'
    )
    parser.add_argument(
        '--no-load',
        action='store_true',
        help='Do not load existing study if it exists'
    )
    
    args = parser.parse_args()
    
    # Create tuner
    tuner = OptunaThroughputTuner(device_num=args.device_num)
    
    # Run optimization
    study = tuner.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=not args.no_load
    )
    
    # Print results
    tuner.print_results(study)
    
    # Export results
    tuner.export_results(study, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
