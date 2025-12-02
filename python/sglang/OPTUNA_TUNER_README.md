# SGLang Optuna Throughput Parameter Tuner

## Overview

The `optuna_throughput_tuner.py` module provides an Optuna-based multi-objective hyperparameter optimization tool for SGLang throughput parameters. It optimizes two objectives simultaneously:

1. **Minimize TTFT** (Time To First Token) - Lower latency for first token generation
2. **Maximize TPS** (Tokens Per Second) - Higher throughput for overall generation

The tuner uses the top 18 most impactful throughput optimization parameters from `top15_throughput_param_generator.py` and applies the same validation rules to ensure only valid parameter combinations are tested.

## Installation

First, install Optuna:

```bash
pip install optuna
```

For database storage support (optional):

```bash
pip install optuna[sqlite]  # For SQLite storage
pip install optuna[postgresql]  # For PostgreSQL storage
```

## Basic Usage

### Quick Start with Default Settings

```bash
python python/sglang/optuna_throughput_tuner.py --n-trials 100
```

This will:
- Run 100 optimization trials
- Use 8 GPUs by default
- Store results in memory only
- Save results to `optuna_results.json`

### Specify Number of GPUs

```bash
python python/sglang/optuna_throughput_tuner.py --n-trials 100 --device-num 4
```

### Persistent Storage with SQLite

```bash
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 200 \
    --storage sqlite:///sglang_optuna.db \
    --study-name my_optimization_study
```

### Resume from Previous Study

```bash
# First run
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 100 \
    --storage sqlite:///sglang_optuna.db \
    --study-name my_study

# Resume and run 100 more trials
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 100 \
    --storage sqlite:///sglang_optuna.db \
    --study-name my_study
```

### Custom Output File

```bash
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 100 \
    --output my_results.json
```

## Understanding the Output

### Console Output

The tuner displays:

1. **Progress Information**: Trial numbers and validation status
2. **Pareto Front Solutions**: Best trade-off configurations between TTFT and TPS
3. **Parameter Details**: Full configuration for each Pareto optimal solution

Example output:

```
================================================================================
Optimization Results
================================================================================

Number of finished trials: 100
Number of Pareto optimal solutions: 5

Pareto Front (Best Trade-offs between TTFT and TPS):
--------------------------------------------------------------------------------

Solution 1:
  TTFT: 0.0250 seconds
  TPS: 2500.00 tokens/sec
  Parameters:
    tp_size: 1
    attention_backend: flashinfer
    ...
```

### JSON Output File

The output JSON file contains:

```json
{
  "study_name": "sglang_throughput_optimization",
  "n_trials": 100,
  "pareto_front": [
    {
      "trial_number": 42,
      "ttft": 0.025,
      "tps": 2500.0,
      "parameters": {
        "tp_size": 1,
        "attention_backend": "flashinfer",
        ...
      },
      "state": "COMPLETE"
    }
  ]
}
```

## Integrating with Real Benchmarks

The default implementation uses a placeholder `run_test()` function. To use with real benchmarks:

### Option 1: Modify the `run_test()` Function

Edit the `run_test()` function in `optuna_throughput_tuner.py`:

```python
def run_test(params: Dict[str, Any]) -> Tuple[float, float]:
    """Run actual SGLang benchmark with parameters."""
    import subprocess
    import json
    
    # Start SGLang server with parameters
    server_process = start_sglang_server(**params)
    
    # Run benchmark
    result = subprocess.run(
        ['python', 'bench_serving.py', '--backend', 'sglang'],
        capture_output=True,
        text=True
    )
    
    # Parse results
    results = json.loads(result.stdout)
    ttft = results['mean_ttft_ms'] / 1000.0  # Convert to seconds
    tps = results['request_throughput']
    
    # Cleanup
    server_process.terminate()
    
    return ttft, tps
```

### Option 2: Use as a Library

```python
from sglang.optuna_throughput_tuner import OptunaThroughputTuner
import optuna

# Create custom tuner
class MyCustomTuner(OptunaThroughputTuner):
    def objective(self, trial: optuna.Trial):
        params = self._suggest_parameters(trial)
        
        if not self._validate_parameters(params):
            raise optuna.TrialPruned("Invalid parameter combination")
        
        # Your custom benchmark code here
        ttft, tps = my_benchmark_function(params)
        
        return ttft, -tps

# Run optimization
tuner = MyCustomTuner(device_num=8)
study = tuner.optimize(n_trials=100)
tuner.print_results(study)
```

## Parameter Search Space

The tuner optimizes these 18 parameters:

| Parameter | Values |
|-----------|--------|
| `tp_size` | 1, 2, 4, 8 |
| `attention_backend` | None, flashinfer, triton, torch_native, fa3, fa4 |
| `chunked_prefill_size` | None, 512, 1024, 2048, 4096, 8192, 16384 |
| `max_prefill_tokens` | 4096, 8192, 16384, 32768, 65536 |
| `dp_size` | 1, 2, 4, 8 |
| `schedule_policy` | fcfs, lpm, random, dfs-weight, lof |
| `pp_size` | 1, 2, 4, 8 |
| `decode_attention_backend` | None, flashinfer, triton, torch_native, fa3 |
| `prefill_attention_backend` | None, flashinfer, triton, torch_native, fa3 |
| `page_size` | None, 16, 32, 64, 128, 256 |
| `cuda_graph_max_bs` | None, 8, 16, 24, 32, 48, 64, 80, 96 |
| `enable_mixed_chunk` | False, True |
| `disable_overlap_schedule` | False, True |
| `enable_torch_compile` | False, True |
| `num_continuous_decode_steps` | 1, 2, 4, 8, 16 |
| `enable_two_batch_overlap` | False, True |
| `tokenizer_worker_num` | 1, 2, 4, 8, 16 |
| `sampling_backend` | None, flashinfer, pytorch |

## Validation Rules

The tuner automatically validates parameter combinations to avoid invalid configurations:

1. **Parallelism Constraint**: `tp_size * pp_size * dp_size ≤ device_num`
2. **Prefill Size Constraint**: `chunked_prefill_size ≤ max_prefill_tokens`
3. **Overlap Scheduling**: Cannot enable overlap features when `disable_overlap_schedule` is True
4. **CUDA Graph Sizing**: Prevents unreasonable batch sizes with small chunk sizes
5. And more... (see code for full details)

## Advanced Usage

### Using Different Samplers

```python
import optuna
from sglang.optuna_throughput_tuner import OptunaThroughputTuner

tuner = OptunaThroughputTuner(device_num=8)

# Use TPE sampler instead of NSGA-II
study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=optuna.samplers.TPESampler()
)

study.optimize(tuner.objective, n_trials=100)
```

### Visualizing Results with Optuna Dashboard

```bash
# Install optuna-dashboard
pip install optuna-dashboard

# Start dashboard
optuna-dashboard sqlite:///sglang_optuna.db
```

Then open http://localhost:8080 in your browser.

### Parallel Optimization

```bash
# Terminal 1
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 50 \
    --storage sqlite:///sglang_optuna.db \
    --study-name parallel_study

# Terminal 2 (run simultaneously)
python python/sglang/optuna_throughput_tuner.py \
    --n-trials 50 \
    --storage sqlite:///sglang_optuna.db \
    --study-name parallel_study
```

## Tips for Effective Optimization

1. **Start Small**: Begin with 50-100 trials to get a sense of the search space
2. **Use Persistent Storage**: Always use `--storage` for long-running optimizations
3. **Monitor Progress**: Check intermediate results to ensure the optimization is progressing
4. **Multiple Runs**: Run multiple optimization sessions to explore different regions
5. **Analyze Pareto Front**: Look at the trade-offs between TTFT and TPS to choose the best configuration for your use case

## Troubleshooting

### Many Trials Being Pruned

If you see many trials being pruned due to "Invalid parameter combination":
- This is normal and expected due to validation constraints
- Reduce `device_num` if you have fewer GPUs
- The sampler will learn to avoid invalid combinations over time

### Out of Memory Errors

If benchmarks fail due to OOM:
- Reduce the maximum values for memory-intensive parameters
- Add custom validation to skip large configurations
- Monitor GPU memory usage during optimization

### Slow Optimization

If optimization is too slow:
- Use parallel optimization with database storage
- Reduce the number of parameters being optimized
- Use a faster sampler like RandomSampler for initial exploration

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SGLang Top 15 Parameters](./top15_throughput_param_generator.py)
- [Multi-Objective Optimization Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
