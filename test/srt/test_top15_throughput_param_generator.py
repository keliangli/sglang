"""Tests for the top 15 throughput parameter combination generator."""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the python directory to the path for direct import
test_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(test_dir, '..', '..'))
python_dir = os.path.join(repo_root, 'python')
sys.path.insert(0, python_dir)

# Import directly from the module file
import importlib.util
module_path = os.path.join(python_dir, 'sglang', 'top15_throughput_param_generator.py')
spec = importlib.util.spec_from_file_location(
    "top15_throughput_param_generator",
    module_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ParameterDefinition = module.ParameterDefinition
Top15ThroughputParamGenerator = module.Top15ThroughputParamGenerator


class TestTop15ThroughputParamGenerator(unittest.TestCase):
    """Test Top15ThroughputParamGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = Top15ThroughputParamGenerator()

    def test_parameter_count(self):
        """Test that exactly 18 parameters are present."""
        self.assertEqual(len(self.generator.parameters), 18,
                        f"Expected 18 parameters, got {len(self.generator.parameters)}")

    def test_top15_parameters_defined(self):
        """Test that the top 18 most impactful parameters are properly defined."""
        param_names = [p.name for p in self.generator.parameters]
        expected_params = [
            "tp_size",
            "attention_backend",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "dp_size",
            "schedule_policy",
            "pp_size",
            "decode_attention_backend",
            "prefill_attention_backend",
            "page_size",
            "cuda_graph_max_bs",
            "enable_mixed_chunk",
            "disable_overlap_schedule",
            "enable_torch_compile",
            "num_continuous_decode_steps",
            "enable_two_batch_overlap",
            "tokenizer_worker_num",
            "sampling_backend",
        ]
        
        self.assertEqual(len(param_names), len(expected_params))
        for expected in expected_params:
            self.assertIn(expected, param_names, f"Parameter {expected} not found")

    def test_generate_combinations_basic(self):
        """Test basic combination generation."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=False,
            max_combinations=10
        )
        self.assertLessEqual(len(combinations), 10)
        
        # Verify each combination has all 18 parameters
        for combo in combinations:
            self.assertEqual(len(combo), 18,
                           f"Combination should have 18 parameters, got {len(combo)}")

    def test_generate_combinations_with_filtering(self):
        """Test combination generation with conflict filtering."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=100
        )
        
        # Verify all combinations are valid
        for combo in combinations:
            self.assertTrue(
                self.generator._is_valid_combination(combo),
                f"Generated invalid combination: {combo}"
            )

    def test_chunked_prefill_constraint(self):
        """Test chunked_prefill_size <= max_prefill_tokens constraint."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=100
        )
        
        for combo in combinations:
            chunked = combo.get("chunked_prefill_size")
            max_prefill = combo.get("max_prefill_tokens")
            if chunked is not None and max_prefill is not None:
                self.assertLessEqual(chunked, max_prefill,
                                   f"Invalid combination: {combo}")

    def test_overlap_schedule_conflicts(self):
        """Test overlap scheduling conflict rules."""
        # disable_overlap_schedule conflicts with enable_two_batch_overlap
        invalid_combo = {
            "disable_overlap_schedule": True,
            "enable_two_batch_overlap": True,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))
        
        # Valid combination
        valid_combo = {
            "disable_overlap_schedule": False,
            "enable_two_batch_overlap": True,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))

    def test_cuda_graph_size_constraints(self):
        """Test CUDA graph batch size constraints."""
        # Large cuda_graph_max_bs with small chunked_prefill_size is invalid
        invalid_combo = {
            "chunked_prefill_size": 1024,
            "cuda_graph_max_bs": 96,
            "max_prefill_tokens": 8192,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))
        
        # Reasonable combination
        valid_combo = {
            "chunked_prefill_size": 8192,
            "cuda_graph_max_bs": 64,
            "max_prefill_tokens": 16384,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))

    def test_max_combinations_limit(self):
        """Test max_combinations parameter."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=10
        )
        self.assertLessEqual(len(combinations), 10)

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        info = self.generator.get_parameter_info()
        
        self.assertIsInstance(info, dict)
        self.assertEqual(len(info), 18)
        
        # Check structure of info
        for param_name, param_info in info.items():
            self.assertIn("values", param_info)
            self.assertIn("description", param_info)
            self.assertIn("num_values", param_info)
            self.assertIn("conflicts_with", param_info)
            self.assertIsInstance(param_info["values"], list)
            self.assertIsInstance(param_info["num_values"], int)
            self.assertGreater(param_info["num_values"], 0)

    def test_export_to_json(self):
        """Test JSON export functionality."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=5
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.generator.export_to_json(temp_path, combinations)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data), len(combinations))
            self.assertEqual(loaded_data, combinations)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_to_csv(self):
        """Test CSV export functionality."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=5
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            self.generator.export_to_csv(temp_path, combinations)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Read and verify CSV content
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
            
            self.assertEqual(len(csv_data), len(combinations))
            
            # Verify headers match
            if combinations:
                expected_headers = set(combinations[0].keys())
                actual_headers = set(csv_data[0].keys())
                self.assertEqual(expected_headers, actual_headers)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_all_combinations_are_valid(self):
        """Test that all generated combinations pass validation."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=100
        )
        
        for combo in combinations:
            self.assertTrue(
                self.generator._is_valid_combination(combo),
                f"Generated invalid combination: {combo}"
            )

    def test_attention_backend_values(self):
        """Test that attention_backend has expected values."""
        param = next(p for p in self.generator.parameters if p.name == "attention_backend")
        expected_values = [None, "flashinfer", "triton", "torch_native", "fa3", "fa4"]
        self.assertEqual(param.values, expected_values)

    def test_schedule_policy_values(self):
        """Test that schedule_policy has expected values."""
        param = next(p for p in self.generator.parameters if p.name == "schedule_policy")
        expected_values = ["fcfs", "lpm", "random", "dfs-weight", "lof"]
        self.assertEqual(param.values, expected_values)

    def test_parameter_order(self):
        """Test that parameters are in the expected order (by impact)."""
        param_names = [p.name for p in self.generator.parameters]
        expected_order = [
            "tp_size",
            "attention_backend",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "dp_size",
            "schedule_policy",
            "pp_size",
            "decode_attention_backend",
            "prefill_attention_backend",
            "page_size",
            "cuda_graph_max_bs",
            "enable_mixed_chunk",
            "disable_overlap_schedule",
            "enable_torch_compile",
            "num_continuous_decode_steps",
            "enable_two_batch_overlap",
            "tokenizer_worker_num",
            "sampling_backend",
        ]
        self.assertEqual(param_names, expected_order)

    def test_add_parameter(self):
        """Test adding a custom parameter."""
        initial_count = len(self.generator.parameters)
        
        new_param = ParameterDefinition(
            name="custom_param",
            values=["x", "y", "z"],
            description="Custom test parameter"
        )
        self.generator.add_parameter(new_param)
        
        self.assertEqual(len(self.generator.parameters), initial_count + 1)
        param_names = [p.name for p in self.generator.parameters]
        self.assertIn("custom_param", param_names)

    def test_remove_parameter(self):
        """Test removing a parameter."""
        initial_count = len(self.generator.parameters)
        
        # Remove a parameter that exists
        self.generator.remove_parameter("schedule_policy")
        
        self.assertEqual(len(self.generator.parameters), initial_count - 1)
        param_names = [p.name for p in self.generator.parameters]
        self.assertNotIn("schedule_policy", param_names)

    def test_parallelism_sizes_exist(self):
        """Test that parallelism size parameters exist."""
        param_names = [p.name for p in self.generator.parameters]
        self.assertIn("tp_size", param_names)
        self.assertIn("pp_size", param_names)
        self.assertIn("dp_size", param_names)

    def test_parallelism_size_values(self):
        """Test that parallelism sizes have reasonable values."""
        tp_param = next(p for p in self.generator.parameters if p.name == "tp_size")
        pp_param = next(p for p in self.generator.parameters if p.name == "pp_size")
        dp_param = next(p for p in self.generator.parameters if p.name == "dp_size")
        
        expected_values = [1, 2, 4, 8]
        self.assertEqual(tp_param.values, expected_values)
        self.assertEqual(pp_param.values, expected_values)
        self.assertEqual(dp_param.values, expected_values)

    def test_parallelism_product_constraint(self):
        """Test that tp_size * pp_size * dp_size <= 64."""
        # Valid combination: 2 * 2 * 2 = 8
        valid_combo = {
            "tp_size": 2,
            "pp_size": 2,
            "dp_size": 2,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))
        
        # Invalid combination: 8 * 8 * 8 = 512 > 64
        invalid_combo = {
            "tp_size": 8,
            "pp_size": 8,
            "dp_size": 8,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))

    def test_parallelism_with_pipeline_constraint(self):
        """Test that pp_size > 4 with dp_size > 4 is invalid."""
        # Invalid: both pp_size and dp_size > 4
        invalid_combo = {
            "tp_size": 1,
            "pp_size": 8,
            "dp_size": 8,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))
        
        # Valid: pp_size > 4 but dp_size <= 4
        valid_combo = {
            "tp_size": 1,
            "pp_size": 8,
            "dp_size": 2,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))

    def test_dp_size_chunked_prefill_constraint(self):
        """Test chunked_prefill_size constraint with dp_size."""
        # Invalid: dp_size=4, chunked_prefill=512 -> effective=128 < 256
        invalid_combo = {
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 4,
            "chunked_prefill_size": 512,
            "max_prefill_tokens": 4096,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))
        
        # Valid: dp_size=2, chunked_prefill=2048 -> effective=1024 >= 256
        valid_combo = {
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 2,
            "chunked_prefill_size": 2048,
            "max_prefill_tokens": 4096,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))

    def test_dp_size_cuda_graph_constraint(self):
        """Test CUDA graph batch size constraint with dp_size."""
        # Invalid: dp_size=4, cuda_graph_max_bs=96
        invalid_combo = {
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 4,
            "cuda_graph_max_bs": 96,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))
        
        # Valid: dp_size=4, cuda_graph_max_bs=32
        valid_combo = {
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 4,
            "cuda_graph_max_bs": 32,
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))


if __name__ == "__main__":
    unittest.main()
