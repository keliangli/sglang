# SGLang Parameter Optimization Guide

## Overview

This guide explains how to use the SGLang throughput parameter combination generator to search for optimal parameter configurations for your workload.

## Parameter Coverage

The generator now supports all 33 critical performance parameters:

### Tokenizer Parameters
- `tokenizer_worker_num`: Number of tokenizer workers (1, 2, 4, 8, 16)

### Memory and Scheduling Parameters
- `chunked_prefill_size`: Chunk size for prefill (None, 512, 1024, 2048, 4096, 8192, 16384)
- `max_prefill_tokens`: Maximum prefill tokens (4096, 8192, 16384, 32768, 65536)
- `schedule_policy`: Request scheduling policy (fcfs, lpm, random, dfs-weight, lof)

### Cache Parameters
- `page_size`: KV cache page size (None, 16, 32, 64, 128, 256)
- `swa_full_tokens_ratio`: SWA token ratio (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
- `radix_eviction_policy`: Radix tree eviction (lru, lfu)

### Attention Backend Parameters
- `attention_backend`: Main attention backend (None, flashinfer, triton, torch_native, fa3, fa4)
- `decode_attention_backend`: Decode attention (None, flashinfer, triton, torch_native, fa3)
- `prefill_attention_backend`: Prefill attention (None, flashinfer, triton, torch_native, fa3)
- `sampling_backend`: Sampling kernels (None, flashinfer, pytorch)
- `mm_attention_backend`: Multimodal attention (None, sdpa, fa3, triton_attn)
- `nsa_prefill_backend`: NSA prefill (flashmla_sparse, flashmla_kv, flashmla_auto, fa3, tilelang)
- `nsa_decode_backend`: NSA decode (fa3, flashmla_kv, flashmla_sparse)

### Hierarchical Cache Parameters
- `enable_hierarchical_cache`: Enable hierarchical caching (False, True)
- `hicache_ratio`: Host/device cache ratio (1.0, 1.5, 2.0, 3.0, 4.0)
- `hicache_size`: Host cache size in GB (0, 8, 16, 32, 64)
- `hicache_write_policy`: Write policy (write_back, write_through, write_through_selective)
- `hicache_io_backend`: IO backend (kernel, direct)
- `hicache_mem_layout`: Memory layout (layer_first, page_first, page_first_direct, page_head)
- `hicache_storage_backend`: Storage backend (None, file, mooncake)
- `hicache_storage_prefetch_policy`: Prefetch policy (best_effort, wait_complete, timeout)

### Cache Options
- `enable_lmcache`: Use LMCache (False, True)
- `disable_radix_cache`: Disable radix cache (False, True)

### CUDA Graph Parameters
- `cuda_graph_max_bs`: Max CUDA graph batch size (None, 8, 16, 24, 32, 48, 64, 80, 96)

### Overlap and Optimization Parameters
- `disable_overlap_schedule`: Disable overlap (False, True)
- `enable_mixed_chunk`: Mixed chunk mode (False, True)
- `enable_two_batch_overlap`: Two batch overlap (False, True)
- `enable_single_batch_overlap`: Single batch overlap (False, True)

### Torch Compile Parameters
- `enable_torch_compile`: Enable torch.compile (False, True)
- `enable_piecewise_cuda_graph`: Piecewise CUDA graph (False, True)
- `torch_compile_max_bs`: Torch compile batch size (16, 32, 48, 64, 96)

### Decode Parameters
- `num_continuous_decode_steps`: Continuous decode steps (1, 2, 4, 8, 16)

## Conflict Detection

The generator automatically filters out invalid parameter combinations based on these rules:

1. **Chunked Prefill Constraint**: `chunked_prefill_size` must be ≤ `max_prefill_tokens`

2. **Hierarchical Cache Conflicts**: 
   - Cannot enable both `enable_hierarchical_cache` and `enable_lmcache`
   - Cannot enable both `enable_hierarchical_cache` and `disable_radix_cache`
   - When `enable_hierarchical_cache` is False, only default hicache parameter values are included to reduce search space

3. **NSA Backend Filtering**: 
   - NSA backends are only varied when `attention_backend` is "nsa"
   - When not using NSA, only default NSA backend values are included to reduce search space

4. **Overlap Scheduling**: Cannot enable overlap features when `disable_overlap_schedule` is True

5. **Memory Layout**: `page_first_direct` layout requires `direct` IO backend

6. **CUDA Graph Sizing**: Prevents unreasonable large batch sizes with small chunk sizes

7. **Torch Compile Filtering**:
   - `torch_compile_max_bs` is only varied when `enable_torch_compile` is True
   - Otherwise, only the default value is included to reduce search space

8. **Radix Cache Filtering**:
   - `radix_eviction_policy` cannot be set when `disable_radix_cache` is True
   - This prevents setting a policy for a disabled feature

These filtering rules significantly reduce the search space from over 10^18 theoretical combinations to a much smaller set of valid, meaningful configurations.

## Usage Examples

### Generate a Small Sample
```bash
python python/sglang/throughput_param_generator.py --max-combinations 100 --output configs.json
```

### Export to CSV
```bash
python python/sglang/throughput_param_generator.py --max-combinations 100 --format csv --output configs.csv
```

### View Parameter Information
```bash
python python/sglang/throughput_param_generator.py --show-info
```

### Disable Conflict Filtering (Not Recommended)
```bash
python python/sglang/throughput_param_generator.py --no-filter --max-combinations 100 --output configs.json
```

## Integration with Benchmarking

See `examples/throughput_param_benchmark_example.py` for an example of how to:
1. Generate parameter combinations
2. Run benchmarks with each configuration
3. Analyze results to find the optimal configuration

## Performance Considerations

The total search space is extremely large (over 10^18 combinations). We recommend:

1. **Start Small**: Generate a sample of 100-1000 configurations
2. **Iterative Refinement**: Analyze results and narrow parameter ranges
3. **Domain Knowledge**: Use your understanding of your workload to constrain parameters
4. **Resource Limits**: Consider your hardware constraints when setting parameters

## Customization

You can customize the parameter generator:

```python
from sglang.throughput_param_generator import ThroughputParamGenerator, ParameterDefinition

gen = ThroughputParamGenerator()

# Remove parameters you don't want to test
gen.remove_parameter("enable_torch_compile")

# Add custom parameters
gen.add_parameter(ParameterDefinition(
    name="custom_param",
    values=[1, 2, 3],
    description="My custom parameter"
))

# Generate combinations
combinations = gen.generate_combinations(max_combinations=100)
```

## Tips for Effective Parameter Search

1. **Focus on High-Impact Parameters**: Start with parameters that have the most impact on your workload
2. **Use Representative Workloads**: Test with data similar to production
3. **Monitor Resources**: Track memory usage, GPU utilization, and throughput
4. **Incremental Testing**: Start with default values and adjust one parameter at a time
5. **Document Results**: Keep track of which configurations work best for different scenarios

## Troubleshooting

### Too Many Combinations
If the search space is too large, consider:
- Reducing the number of values per parameter
- Removing less critical parameters
- Using the `--max-combinations` flag to limit output

### Invalid Combinations Generated
If you find invalid combinations:
- Check that conflict rules are properly defined
- Report issues with specific parameter combinations
- Use `--no-filter` to see all combinations and manually filter

### Performance Issues
If generation is slow:
- Reduce the number of parameters
- Use `--max-combinations` to generate samples
- Consider generating in batches

## Further Reading

- [SGLang Server Arguments Documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)
- [Benchmarking Guide](../benchmark/README.md)
- [Performance Tuning Guide](./performance_tuning.md)
