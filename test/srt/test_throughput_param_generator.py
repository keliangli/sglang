"""Tests for the throughput parameter combination generator."""

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
module_path = os.path.join(python_dir, 'sglang', 'throughput_param_generator.py')
spec = importlib.util.spec_from_file_location(
    "throughput_param_generator",
    module_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ParameterDefinition = module.ParameterDefinition
ThroughputParamGenerator = module.ThroughputParamGenerator


class TestParameterDefinition(unittest.TestCase):
    """Test ParameterDefinition class."""

    def test_basic_creation(self):
        """Test basic parameter definition creation."""
        param = ParameterDefinition(
            name="test_param",
            values=[1, 2, 3],
            description="Test parameter"
        )
        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.values, [1, 2, 3])
        self.assertEqual(param.description, "Test parameter")
        self.assertEqual(len(param.conflicts_with), 0)

    def test_with_conflicts(self):
        """Test parameter definition with conflicts."""
        param = ParameterDefinition(
            name="param1",
            values=[True, False],
            conflicts_with={("param2", "value")}
        )
        self.assertIn(("param2", "value"), param.conflicts_with)

    def test_conflicts_list_conversion(self):
        """Test that conflicts list is converted to set."""
        param = ParameterDefinition(
            name="param1",
            values=[1, 2],
            conflicts_with=[("param2", "value")]
        )
        self.assertIsInstance(param.conflicts_with, set)


class TestThroughputParamGenerator(unittest.TestCase):
    """Test ThroughputParamGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ThroughputParamGenerator()

    def test_default_parameters_defined(self):
        """Test that default parameters are properly defined."""
        self.assertGreater(len(self.generator.parameters), 0)
        
        # Check that key parameters exist
        param_names = [p.name for p in self.generator.parameters]
        expected_params = [
            "tokenizer_worker_num",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "schedule_policy",
            "page_size",
            "swa_full_tokens_ratio",
            "radix_eviction_policy",
        ]
        for expected in expected_params:
            self.assertIn(expected, param_names)

    def test_generate_combinations_basic(self):
        """Test basic combination generation."""
        # Create a simple generator with few parameters
        simple_gen = ThroughputParamGenerator()
        simple_gen.parameters = [
            ParameterDefinition(name="param1", values=[1, 2]),
            ParameterDefinition(name="param2", values=["a", "b"]),
        ]
        
        combinations = simple_gen.generate_combinations(filter_conflicts=False)
        # Should generate 2 * 2 = 4 combinations
        self.assertEqual(len(combinations), 4)
        
        # Verify all combinations are present
        expected = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        for exp in expected:
            self.assertIn(exp, combinations)

    def test_generate_combinations_with_filtering(self):
        """Test combination generation with conflict filtering."""
        simple_gen = ThroughputParamGenerator()
        simple_gen.parameters = [
            ParameterDefinition(name="param1", values=[True, False]),
            ParameterDefinition(
                name="param2",
                values=[1, 2],
                conflicts_with={("param1", True)}
            ),
        ]
        
        combinations = simple_gen.generate_combinations(filter_conflicts=True)
        
        # Should filter out combinations where param1=True and param2 has a value
        for combo in combinations:
            if combo["param1"] is True:
                # When param1 is True, param2 should not have values that conflict
                # But our conflict is defined on param2 side, so all param2 values conflict with param1=True
                # This means no combinations with param1=True should remain
                self.fail(f"Found invalid combination: {combo}")

    def test_chunked_prefill_constraint(self):
        """Test chunked_prefill_size <= max_prefill_tokens constraint."""
        combinations = self.generator.generate_combinations(filter_conflicts=True)
        
        for combo in combinations:
            chunked = combo.get("chunked_prefill_size")
            max_prefill = combo.get("max_prefill_tokens")
            if chunked is not None and max_prefill is not None:
                self.assertLessEqual(chunked, max_prefill,
                                   f"Invalid combination: {combo}")

    def test_max_combinations_limit(self):
        """Test max_combinations parameter."""
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=10
        )
        self.assertLessEqual(len(combinations), 10)

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

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        info = self.generator.get_parameter_info()
        
        self.assertIsInstance(info, dict)
        self.assertGreater(len(info), 0)
        
        # Check structure of info
        for param_name, param_info in info.items():
            self.assertIn("values", param_info)
            self.assertIn("description", param_info)
            self.assertIn("num_values", param_info)
            self.assertIn("conflicts_with", param_info)
            self.assertIsInstance(param_info["values"], list)
            self.assertIsInstance(param_info["num_values"], int)

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
        import csv
        
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

    def test_export_csv_empty_combinations(self):
        """Test CSV export with empty combinations list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            self.generator.export_to_csv(temp_path, [])
            # Should handle empty list gracefully
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_is_valid_combination(self):
        """Test the validation logic for combinations."""
        # Valid combination
        valid_combo = {
            "chunked_prefill_size": 512,
            "max_prefill_tokens": 1024,
            "tokenizer_worker_num": 2,
            "schedule_policy": "fcfs",
        }
        self.assertTrue(self.generator._is_valid_combination(valid_combo))
        
        # Invalid: chunked_prefill > max_prefill
        invalid_combo = {
            "chunked_prefill_size": 2048,
            "max_prefill_tokens": 1024,
        }
        self.assertFalse(self.generator._is_valid_combination(invalid_combo))

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


if __name__ == "__main__":
    unittest.main()
