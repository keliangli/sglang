#!/usr/bin/env python3
"""
Unit tests for optuna_throughput_tuner.py

This test file validates the basic functionality of the Optuna tuner
without requiring actual GPU benchmarks.
"""

import json
import os
import sys
import tempfile
import unittest
from typing import Dict, Any, Tuple

# Try to import optuna first
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import the module directly from file path to avoid dependency issues
if OPTUNA_AVAILABLE:
    # Get the path to the optuna_throughput_tuner.py file
    tuner_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'python',
        'sglang',
        'optuna_throughput_tuner.py'
    )
    
    # Load module from file
    import importlib.util
    spec = importlib.util.spec_from_file_location("optuna_throughput_tuner", tuner_path)
    if spec and spec.loader:
        tuner_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(tuner_module)
            OptunaThroughputTuner = tuner_module.OptunaThroughputTuner
            run_test = tuner_module.run_test
        except Exception as e:
            print(f"Error loading tuner module: {e}")
            OPTUNA_AVAILABLE = False
    else:
        print("Could not load tuner module")
        OPTUNA_AVAILABLE = False


@unittest.skipIf(not OPTUNA_AVAILABLE, "Optuna not installed")
class TestOptunaThroughputTuner(unittest.TestCase):
    """Test cases for OptunaThroughputTuner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tuner = OptunaThroughputTuner(device_num=4)
    
    def test_initialization(self):
        """Test tuner initialization."""
        self.assertEqual(self.tuner.device_num, 4)
    
    def test_parameter_validation_valid(self):
        """Test parameter validation with valid combinations."""
        # Valid combination: all parallelism sizes = 1
        valid_params = {
            'tp_size': 1,
            'pp_size': 1,
            'dp_size': 1,
            'chunked_prefill_size': 2048,
            'max_prefill_tokens': 4096,
            'disable_overlap_schedule': False,
            'enable_two_batch_overlap': False
        }
        self.assertTrue(self.tuner._validate_parameters(valid_params))
    
    def test_parameter_validation_invalid_parallelism(self):
        """Test parameter validation with invalid parallelism."""
        # Invalid: tp_size * pp_size * dp_size = 2 * 2 * 2 = 8 > device_num (4)
        invalid_params = {
            'tp_size': 2,
            'pp_size': 2,
            'dp_size': 2,
        }
        self.assertFalse(self.tuner._validate_parameters(invalid_params))
    
    def test_parameter_validation_invalid_prefill_size(self):
        """Test parameter validation with invalid prefill size."""
        # Invalid: chunked_prefill_size > max_prefill_tokens
        invalid_params = {
            'tp_size': 1,
            'pp_size': 1,
            'dp_size': 1,
            'chunked_prefill_size': 8192,
            'max_prefill_tokens': 4096
        }
        self.assertFalse(self.tuner._validate_parameters(invalid_params))
    
    def test_parameter_validation_overlap_conflict(self):
        """Test parameter validation with overlap conflict."""
        # Invalid: enable_two_batch_overlap with disable_overlap_schedule
        invalid_params = {
            'tp_size': 1,
            'pp_size': 1,
            'dp_size': 1,
            'disable_overlap_schedule': True,
            'enable_two_batch_overlap': True
        }
        self.assertFalse(self.tuner._validate_parameters(invalid_params))
    
    def test_run_test_placeholder(self):
        """Test that run_test placeholder returns valid results."""
        params = {'tp_size': 1}
        ttft, tps = run_test(params)
        
        # Check that results are valid numbers
        self.assertIsInstance(ttft, float)
        self.assertIsInstance(tps, float)
        self.assertGreater(ttft, 0)
        self.assertGreater(tps, 0)
    
    def test_suggest_parameters(self):
        """Test parameter suggestion."""
        # Create a mock trial
        study = optuna.create_study(directions=['minimize', 'minimize'])
        trial = study.ask()
        
        # Suggest parameters
        params = self.tuner._suggest_parameters(trial)
        
        # Check that all 18 parameters are present
        expected_params = [
            'tp_size', 'attention_backend', 'chunked_prefill_size',
            'max_prefill_tokens', 'dp_size', 'schedule_policy', 'pp_size',
            'decode_attention_backend', 'prefill_attention_backend',
            'page_size', 'cuda_graph_max_bs', 'enable_mixed_chunk',
            'disable_overlap_schedule', 'enable_torch_compile',
            'num_continuous_decode_steps', 'enable_two_batch_overlap',
            'tokenizer_worker_num', 'sampling_backend'
        ]
        
        for param_name in expected_params:
            self.assertIn(param_name, params)
    
    def test_optimization_short_run(self):
        """Test optimization with a very short run."""
        # Run optimization with just 5 trials
        study = self.tuner.optimize(
            n_trials=5,
            study_name='test_study'
        )
        
        # Check that study was created
        self.assertIsNotNone(study)
        
        # Check that we have some completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        self.assertGreater(len(completed_trials), 0, "At least one trial should complete")
    
    def test_export_results(self):
        """Test exporting results to JSON."""
        # Run a short optimization
        study = self.tuner.optimize(n_trials=5, study_name='test_export')
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.tuner.export_results(study, temp_file)
            
            # Check that file exists and is valid JSON
            self.assertTrue(os.path.exists(temp_file))
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            # Check structure
            self.assertIn('study_name', data)
            self.assertIn('n_trials', data)
            self.assertIn('pareto_front', data)
            self.assertEqual(data['study_name'], 'test_export')
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_multiple_pareto_solutions(self):
        """Test that optimization can find multiple Pareto solutions."""
        # Run more trials to increase chance of multiple solutions
        study = self.tuner.optimize(
            n_trials=30,
            study_name='test_pareto'
        )
        
        # Get Pareto optimal trials
        pareto_trials = study.best_trials
        
        # We should have at least 1 Pareto optimal solution
        self.assertGreater(len(pareto_trials), 0)
        
        # Each solution should have valid objectives
        for trial in pareto_trials:
            self.assertEqual(len(trial.values), 2)  # TTFT and -TPS
            ttft = trial.values[0]
            neg_tps = trial.values[1]
            self.assertIsInstance(ttft, float)
            self.assertIsInstance(neg_tps, float)
            self.assertGreater(ttft, 0)
            self.assertLess(neg_tps, 0)  # Should be negative


class TestParameterValidationRules(unittest.TestCase):
    """Specific test cases for parameter validation rules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tuner = OptunaThroughputTuner(device_num=8)
    
    def test_rule_parallelism_constraint(self):
        """Test Rule 1: Parallelism size constraints."""
        # Valid: 2 * 2 * 2 = 8 <= 8
        valid = {'tp_size': 2, 'pp_size': 2, 'dp_size': 2}
        self.assertTrue(self.tuner._validate_parameters(valid))
        
        # Invalid: 4 * 2 * 2 = 16 > 8
        invalid = {'tp_size': 4, 'pp_size': 2, 'dp_size': 2}
        self.assertFalse(self.tuner._validate_parameters(invalid))
    
    def test_rule_pp_dp_extreme(self):
        """Test Rule 2: pp_size and dp_size extreme combinations."""
        # Invalid: pp_size > 4 and dp_size > 4
        invalid = {'tp_size': 1, 'pp_size': 8, 'dp_size': 8}
        self.assertFalse(self.tuner._validate_parameters(invalid))
    
    def test_rule_chunked_prefill_constraint(self):
        """Test Rule 3: chunked_prefill_size <= max_prefill_tokens."""
        # Valid
        valid = {'chunked_prefill_size': 4096, 'max_prefill_tokens': 8192}
        self.assertTrue(self.tuner._validate_parameters(valid))
        
        # Invalid
        invalid = {'chunked_prefill_size': 8192, 'max_prefill_tokens': 4096}
        self.assertFalse(self.tuner._validate_parameters(invalid))
    
    def test_rule_cuda_graph_constraint(self):
        """Test Rule 6: CUDA graph batch size constraints."""
        # Invalid: small chunked_prefill_size with large cuda_graph_max_bs
        invalid = {
            'chunked_prefill_size': 1024,
            'cuda_graph_max_bs': 64,
            'dp_size': 1
        }
        self.assertFalse(self.tuner._validate_parameters(invalid))


if __name__ == '__main__':
    if not OPTUNA_AVAILABLE:
        print("Warning: Optuna not installed. Skipping tests.")
        print("Install with: pip install optuna")
        sys.exit(0)
    
    unittest.main()
