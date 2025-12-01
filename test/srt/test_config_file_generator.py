"""Tests for the ConfigFileGenerator class."""

import json
import os
import shutil
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

ConfigFileGenerator = module.ConfigFileGenerator
Top15ThroughputParamGenerator = module.Top15ThroughputParamGenerator


class TestConfigFileGenerator(unittest.TestCase):
    """Test ConfigFileGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = Top15ThroughputParamGenerator()
        self.config_generator = ConfigFileGenerator(self.generator)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ConfigFileGenerator initialization."""
        # Test with provided generator
        gen = ConfigFileGenerator(self.generator)
        self.assertIsNotNone(gen.generator)
        self.assertIsNotNone(gen.default_config_template)
        
        # Test without provided generator (should create one)
        gen2 = ConfigFileGenerator()
        self.assertIsNotNone(gen2.generator)
        self.assertIsInstance(gen2.generator, Top15ThroughputParamGenerator)

    def test_default_template_structure(self):
        """Test that default config template has required fields."""
        template = self.config_generator.default_config_template
        
        required_fields = [
            "context_length",
            "trust_remote_code",
            "dtype",
            "kv_cache_dtype",
            "quantization",
            "quantization_param_path",
            "mem_fraction_static",
            "enable_chunked_prefill",
            "max_running_requests",
            "chunked_prefill_size",
            "random_seed",
            "enable_ep_moe",
            "disable_chunked_prefix_cache"
        ]
        
        for field in required_fields:
            self.assertIn(field, template, f"Required field {field} not in template")

    def test_merge_params_with_template(self):
        """Test merging parameters with template."""
        params = {
            "tp_size": 2,
            "attention_backend": "flashinfer",
            "chunked_prefill_size": 2048,
            "max_prefill_tokens": 8192,
        }
        
        merged = self.config_generator._merge_params_with_template(params)
        
        # Check that template fields are present
        self.assertIn("context_length", merged)
        self.assertIn("dtype", merged)
        
        # Check that parameters are merged
        self.assertIn("tp_size", merged)
        self.assertEqual(merged["tp_size"], 2)
        self.assertEqual(merged["attention_backend"], "flashinfer")
        self.assertEqual(merged["chunked_prefill_size"], 2048)

    def test_merge_params_with_custom_template(self):
        """Test merging parameters with custom template."""
        custom_template = {
            "custom_field": "custom_value",
            "context_length": 8192,
        }
        
        params = {
            "tp_size": 4,
        }
        
        merged = self.config_generator._merge_params_with_template(params, custom_template)
        
        # Check that custom template is used
        self.assertEqual(merged["custom_field"], "custom_value")
        self.assertEqual(merged["context_length"], 8192)
        self.assertEqual(merged["tp_size"], 4)

    def test_save_config_file(self):
        """Test saving a single config file."""
        params = {
            "tp_size": 2,
            "attention_backend": "flashinfer",
            "chunked_prefill_size": 2048,
        }
        
        filepath = os.path.join(self.temp_dir, "test_config.json")
        self.config_generator.save_config_file(params, filepath)
        
        # Verify file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertIn("tp_size", loaded_config)
        self.assertEqual(loaded_config["tp_size"], 2)
        self.assertIn("context_length", loaded_config)
        self.assertIn("dtype", loaded_config)

    def test_save_config_file_creates_directory(self):
        """Test that save_config_file creates directory if needed."""
        nested_dir = os.path.join(self.temp_dir, "nested", "dir")
        filepath = os.path.join(nested_dir, "config.json")
        
        params = {"tp_size": 1}
        self.config_generator.save_config_file(params, filepath)
        
        # Verify file exists and directory was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.path.isdir(nested_dir))

    def test_save_config_file_with_custom_indent(self):
        """Test saving config file with custom indentation."""
        params = {"tp_size": 2}
        filepath = os.path.join(self.temp_dir, "config_indent.json")
        
        # Save with indent=4
        self.config_generator.save_config_file(params, filepath, indent=4)
        
        # Read file content to check indentation
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Should have 4-space indentation
        self.assertIn("    ", content)

    def test_save_multiple_configs(self):
        """Test saving multiple config files."""
        combinations = [
            {"tp_size": 1, "attention_backend": "flashinfer"},
            {"tp_size": 2, "attention_backend": "triton"},
            {"tp_size": 4, "attention_backend": "torch_native"},
        ]
        
        created_files = self.config_generator.save_multiple_configs(
            combinations,
            self.temp_dir
        )
        
        # Verify correct number of files created
        self.assertEqual(len(created_files), 3)
        
        # Verify all files exist
        for filepath in created_files:
            self.assertTrue(os.path.exists(filepath))
        
        # Verify content of first file
        with open(created_files[0], 'r') as f:
            config = json.load(f)
        self.assertEqual(config["tp_size"], 1)

    def test_save_multiple_configs_with_custom_pattern(self):
        """Test saving multiple configs with custom filename pattern."""
        combinations = [
            {"tp_size": 1},
            {"tp_size": 2},
        ]
        
        created_files = self.config_generator.save_multiple_configs(
            combinations,
            self.temp_dir,
            filename_pattern="my_config_{index}.json"
        )
        
        # Verify filenames match pattern
        self.assertTrue(created_files[0].endswith("my_config_0.json"))
        self.assertTrue(created_files[1].endswith("my_config_1.json"))

    def test_save_multiple_configs_empty_list(self):
        """Test saving with empty combinations list."""
        created_files = self.config_generator.save_multiple_configs(
            [],
            self.temp_dir
        )
        
        # Should return empty list
        self.assertEqual(len(created_files), 0)

    def test_generate_and_save_configs(self):
        """Test generating and saving configs in one call."""
        created_files = self.config_generator.generate_and_save_configs(
            self.temp_dir,
            max_combinations=5,
            filter_conflicts=True
        )
        
        # Verify files were created
        self.assertGreater(len(created_files), 0)
        self.assertLessEqual(len(created_files), 5)
        
        # Verify each file is valid JSON
        for filepath in created_files:
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'r') as f:
                config = json.load(f)
                # Should have template fields
                self.assertIn("context_length", config)
                self.assertIn("dtype", config)
                # Should have parameter fields
                self.assertIn("tp_size", config)

    def test_generate_and_save_configs_with_custom_template(self):
        """Test generating configs with custom template."""
        custom_template = {
            "my_custom_field": "custom_value",
            "context_length": 8192,
        }
        
        created_files = self.config_generator.generate_and_save_configs(
            self.temp_dir,
            max_combinations=2,
            template=custom_template
        )
        
        # Verify custom template is used
        with open(created_files[0], 'r') as f:
            config = json.load(f)
        
        self.assertIn("my_custom_field", config)
        self.assertEqual(config["my_custom_field"], "custom_value")

    def test_config_file_content_validity(self):
        """Test that saved config files contain valid data."""
        params = {
            "tp_size": 2,
            "attention_backend": "flashinfer",
            "chunked_prefill_size": 4096,
            "max_prefill_tokens": 8192,
            "dp_size": 1,
            "schedule_policy": "fcfs",
            "pp_size": 1,
        }
        
        filepath = os.path.join(self.temp_dir, "validity_test.json")
        self.config_generator.save_config_file(params, filepath)
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Verify all parameter values are preserved
        self.assertEqual(config["tp_size"], 2)
        self.assertEqual(config["attention_backend"], "flashinfer")
        self.assertEqual(config["chunked_prefill_size"], 4096)
        
        # Verify template defaults are present
        self.assertEqual(config["context_length"], 4096)
        self.assertEqual(config["trust_remote_code"], True)
        self.assertEqual(config["dtype"], "auto")
        self.assertEqual(config["random_seed"], 42)

    def test_save_configs_with_none_values(self):
        """Test saving configs with None values (should be preserved)."""
        params = {
            "attention_backend": None,
            "page_size": None,
        }
        
        filepath = os.path.join(self.temp_dir, "none_values.json")
        self.config_generator.save_config_file(params, filepath)
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Verify None values are preserved as null in JSON
        self.assertIsNone(config["attention_backend"])
        self.assertIsNone(config["page_size"])

    def test_integration_with_generator(self):
        """Test integration with Top15ThroughputParamGenerator."""
        # Generate some combinations
        combinations = self.generator.generate_combinations(
            filter_conflicts=True,
            max_combinations=3
        )
        
        # Save them using config generator
        created_files = self.config_generator.save_multiple_configs(
            combinations,
            self.temp_dir
        )
        
        self.assertEqual(len(created_files), 3)
        
        # Verify each file contains valid merged config
        for filepath in created_files:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Should have both template and parameter fields
            self.assertIn("context_length", config)  # from template
            self.assertIn("tp_size", config)  # from parameters
            self.assertIn("attention_backend", config)  # from parameters


if __name__ == "__main__":
    unittest.main()
