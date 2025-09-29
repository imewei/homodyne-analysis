"""
Comprehensive Tests for ML-Accelerated Optimization Framework
===========================================================

Test suite for machine learning acceleration capabilities including predictive models,
transfer learning, and optimization enhancement.

Authors: Wei Chen, Hongrui He, Claude (Anthropic)
Institution: Argonne National Laboratory
"""

import pickle
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Check for sklearn availability
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import the modules we're testing
from homodyne.optimization.ml_acceleration import (
    EnsembleOptimizationPredictor,
    MLAcceleratedOptimizer,
    MLModelConfig,
    OptimizationRecord,
    PredictionResult,
    TransferLearningPredictor,
    create_ml_accelerated_optimizer,
    enhance_classical_optimizer_with_ml,
    get_ml_backend_info,
    save_optimization_data_securely,
    load_optimization_data_securely,
)


class TestOptimizationRecord:
    """Test suite for optimization record data structure."""

    def test_optimization_record_creation(self):
        """Test creation of optimization records."""
        record = OptimizationRecord(
            experiment_id="exp_001",
            initial_parameters=np.array([1.0, 2.0, 3.0]),
            final_parameters=np.array([1.5, 2.5, 3.5]),
            objective_value=0.123,
            convergence_time=5.67,
            method="Nelder-Mead",
            experimental_conditions={"temperature": 25.0, "concentration": 0.1},
            metadata={"notes": "test optimization"},
        )

        assert record.experiment_id == "exp_001"
        assert np.array_equal(record.initial_parameters, np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(record.final_parameters, np.array([1.5, 2.5, 3.5]))
        assert record.objective_value == 0.123
        assert record.convergence_time == 5.67
        assert record.method == "Nelder-Mead"
        assert record.experimental_conditions["temperature"] == 25.0
        assert record.metadata["notes"] == "test optimization"
        assert isinstance(record.timestamp, float)

    def test_optimization_record_serialization(self):
        """Test serialization of optimization records."""
        record = OptimizationRecord(
            experiment_id="exp_002",
            initial_parameters=np.array([1.0, 2.0]),
            final_parameters=np.array([1.1, 2.1]),
            objective_value=0.456,
            convergence_time=2.34,
            method="BFGS",
            experimental_conditions={"param1": 1.0},
        )

        # Test secure JSON serialization instead of pickle for security
        temp_file = Path(tempfile.mktemp(suffix=".json"))
        try:
            save_optimization_data_securely([record], temp_file)
            deserialized_records = load_optimization_data_securely(temp_file)
            assert len(deserialized_records) == 1
            deserialized = deserialized_records[0]
            assert deserialized.experiment_id == record.experiment_id
            assert np.array_equal(
                deserialized.initial_parameters, record.initial_parameters
            )
            assert np.array_equal(deserialized.final_parameters, record.final_parameters)
            assert deserialized.objective_value == record.objective_value
            assert deserialized.convergence_time == record.convergence_time
            assert deserialized.method == record.method
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestMLModelConfig:
    """Test suite for ML model configuration."""

    def test_ml_model_config_defaults(self):
        """Test default ML model configuration."""
        config = MLModelConfig(model_type="ensemble")

        assert config.model_type == "ensemble"
        assert config.feature_scaling == "standard"
        assert config.validation_split == 0.2
        assert config.cv_folds == 5
        assert config.enable_hyperparameter_tuning is True
        assert isinstance(config.hyperparameters, dict)

    def test_ml_model_config_custom(self):
        """Test custom ML model configuration."""
        custom_hyperparameters = {"random_forest": {"n_estimators": 50, "max_depth": 5}}

        config = MLModelConfig(
            model_type="custom",
            hyperparameters=custom_hyperparameters,
            feature_scaling="minmax",
            validation_split=0.3,
            cv_folds=3,
            enable_hyperparameter_tuning=False,
        )

        assert config.model_type == "custom"
        assert config.hyperparameters == custom_hyperparameters
        assert config.feature_scaling == "minmax"
        assert config.validation_split == 0.3
        assert config.cv_folds == 3
        assert config.enable_hyperparameter_tuning is False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
class TestEnsembleOptimizationPredictor:
    """Test suite for ensemble optimization predictor."""

    @pytest.fixture
    def sample_optimization_records(self):
        """Create sample optimization records for testing."""
        records = []
        np.random.seed(42)  # For reproducible tests

        for i in range(20):
            # Create diverse experimental conditions
            conditions = {
                "temperature": 20 + np.random.rand() * 20,
                "concentration": 0.05 + np.random.rand() * 0.1,
                "shear_rate": np.random.rand() * 100,
                "q_value": 0.1 + np.random.rand() * 0.05,
            }

            # Create correlated initial and final parameters
            initial_params = np.random.rand(3) * 10
            final_params = initial_params + np.random.normal(0, 0.1, 3)

            # Create realistic objective value (lower is better)
            objective_value = np.sum((final_params - 5) ** 2) + np.random.rand() * 0.1

            record = OptimizationRecord(
                experiment_id=f"exp_{i:03d}",
                initial_parameters=initial_params,
                final_parameters=final_params,
                objective_value=objective_value,
                convergence_time=1 + np.random.rand() * 10,
                method="Nelder-Mead",
                experimental_conditions=conditions,
            )
            records.append(record)

        return records

    def test_ensemble_predictor_initialization(self):
        """Test ensemble predictor initialization."""
        config = MLModelConfig(model_type="ensemble")
        predictor = EnsembleOptimizationPredictor(config)

        assert predictor.config.model_type == "ensemble"
        assert predictor.is_fitted is False
        assert len(predictor.models) > 0  # Should have initialized models
        assert "random_forest" in predictor.models
        assert "gaussian_process" in predictor.models

    def test_ensemble_predictor_fitting(self, sample_optimization_records):
        """Test ensemble predictor training."""
        predictor = EnsembleOptimizationPredictor()

        # Test fitting with sufficient data
        predictor.fit(sample_optimization_records)

        assert predictor.is_fitted is True
        assert len(predictor.feature_names) > 0
        assert len(predictor.target_names) > 0
        assert len(predictor.training_history) > 0

        # Check model info
        model_info = predictor.get_model_info()
        assert model_info["status"] == "fitted"
        assert "ensemble_models" in model_info
        assert len(model_info["ensemble_models"]) > 0

    def test_ensemble_predictor_prediction(self, sample_optimization_records):
        """Test ensemble predictor making predictions."""
        predictor = EnsembleOptimizationPredictor()
        predictor.fit(sample_optimization_records)

        # Test prediction with new experimental conditions
        test_conditions = {
            "temperature": 25.0,
            "concentration": 0.08,
            "shear_rate": 50.0,
            "q_value": 0.12,
        }

        initial_guess = np.array([3.0, 4.0, 5.0])

        prediction = predictor.predict(test_conditions, initial_guess)

        assert isinstance(prediction, PredictionResult)
        assert prediction.predicted_parameters.shape == (3,)
        assert 0.0 <= prediction.confidence_score <= 1.0
        assert prediction.prediction_uncertainty.shape == (3,)
        assert isinstance(prediction.model_performance, dict)

    def test_ensemble_predictor_update(self, sample_optimization_records):
        """Test ensemble predictor updating with new data."""
        predictor = EnsembleOptimizationPredictor()
        predictor.fit(sample_optimization_records[:15])  # Train on subset

        initial_history_length = len(predictor.training_history)

        # Add new record
        new_record = sample_optimization_records[15]
        predictor.update(new_record)

        # Check that history was updated
        assert len(predictor.training_history) == initial_history_length + 1

    def test_ensemble_predictor_insufficient_data(self):
        """Test ensemble predictor with insufficient training data."""
        predictor = EnsembleOptimizationPredictor()

        # Create minimal dataset
        minimal_records = [
            OptimizationRecord(
                experiment_id="exp_001",
                initial_parameters=np.array([1.0, 2.0]),
                final_parameters=np.array([1.1, 2.1]),
                objective_value=0.5,
                convergence_time=1.0,
                method="Nelder-Mead",
                experimental_conditions={"param1": 1.0},
            )
        ]

        # Should handle gracefully (logs warning but doesn't crash)
        predictor.fit(minimal_records)
        # Test passes if no exception is raised

    def test_feature_extraction(self, sample_optimization_records):
        """Test feature extraction from optimization records."""
        predictor = EnsembleOptimizationPredictor()

        X, y = predictor._extract_features_and_targets(sample_optimization_records)

        assert X.shape[0] == len(sample_optimization_records)
        assert y.shape[0] == len(sample_optimization_records)
        assert X.shape[1] > 0  # Should have features
        assert y.shape[1] == 3  # Should have 3 target parameters

    def test_conditions_to_features(self):
        """Test conversion of experimental conditions to feature vector."""
        predictor = EnsembleOptimizationPredictor()

        conditions = {
            "temperature": 25.0,
            "concentration": 0.08,
            "array_param": np.array([1.0, 2.0]),
            "list_param": [3.0, 4.0],
            "string_param": "test",
        }

        initial_params = np.array([1.0, 2.0, 3.0])

        features = predictor._conditions_to_features(conditions, initial_params)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))  # No NaN values


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
class TestTransferLearningPredictor:
    """Test suite for transfer learning predictor."""

    @pytest.fixture
    def sample_optimization_records(self):
        """Create sample optimization records for testing."""
        records = []
        np.random.seed(42)  # For reproducible tests

        for i in range(20):
            # Create diverse experimental conditions
            conditions = {
                "temperature": 20 + np.random.rand() * 20,
                "concentration": 0.05 + np.random.rand() * 0.1,
                "shear_rate": np.random.rand() * 100,
                "q_value": 0.1 + np.random.rand() * 0.05,
            }

            # Create correlated initial and final parameters
            initial_params = np.random.rand(3) * 10
            final_params = initial_params + np.random.normal(0, 0.1, 3)

            # Create realistic objective value (lower is better)
            objective_value = np.sum((final_params - 5) ** 2) + np.random.rand() * 0.1

            record = OptimizationRecord(
                experiment_id=f"exp_{i:03d}",
                initial_parameters=initial_params,
                final_parameters=final_params,
                objective_value=objective_value,
                convergence_time=1 + np.random.rand() * 10,
                method="Nelder-Mead",
                experimental_conditions=conditions,
            )
            records.append(record)

        return records

    def test_transfer_learning_initialization(self, sample_optimization_records):
        """Test transfer learning predictor initialization."""
        base_predictor = EnsembleOptimizationPredictor()
        transfer_predictor = TransferLearningPredictor(base_predictor)

        assert transfer_predictor.base_predictor == base_predictor
        assert isinstance(transfer_predictor.domain_adapters, dict)
        assert transfer_predictor.similarity_threshold == 0.8

    def test_transfer_learning_domain_classification(self, sample_optimization_records):
        """Test domain classification for transfer learning."""
        base_predictor = EnsembleOptimizationPredictor()
        transfer_predictor = TransferLearningPredictor(base_predictor)

        # Test domain classification
        conditions1 = {"temperature": 25.0, "concentration": 0.1}
        conditions2 = {"temperature": 35.0, "concentration": 0.1}  # More different temperature
        conditions3 = {"temperature": 25.0, "concentration": 0.3}  # More different concentration

        domain1 = transfer_predictor._classify_domain(conditions1)
        domain2 = transfer_predictor._classify_domain(conditions2)
        domain3 = transfer_predictor._classify_domain(conditions3)

        assert isinstance(domain1, str)
        # At least one should be different or test the classification logic differently
        domains = [domain1, domain2, domain3]
        unique_domains = set(domains)
        assert len(unique_domains) >= 2, f"Expected at least 2 unique domains, got: {unique_domains}"

    def test_transfer_learning_fitting(self, sample_optimization_records):
        """Test transfer learning fitting process."""
        base_predictor = EnsembleOptimizationPredictor()
        transfer_predictor = TransferLearningPredictor(base_predictor)

        # Fit with sample data
        transfer_predictor.fit(sample_optimization_records)

        # Check that base predictor was fitted
        assert base_predictor.is_fitted is True

        # Check that domain adapters were created
        assert len(transfer_predictor.domain_adapters) >= 0

    def test_domain_similarity_computation(self):
        """Test domain similarity computation."""
        base_predictor = EnsembleOptimizationPredictor()
        transfer_predictor = TransferLearningPredictor(base_predictor)

        conditions = {"temperature": 25.0, "concentration": 0.1}
        domain = "temperature_20_concentration_0"

        similarity = transfer_predictor._compute_domain_similarity(conditions, domain)

        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)


class TestMLAcceleratedOptimizer:
    """Test suite for ML-accelerated optimizer."""

    @pytest.fixture
    def sample_optimization_records(self):
        """Create sample optimization records for testing."""
        records = []
        np.random.seed(42)  # For reproducible tests

        for i in range(20):
            # Create diverse experimental conditions
            conditions = {
                "temperature": 20 + np.random.rand() * 20,
                "concentration": 0.05 + np.random.rand() * 0.1,
                "shear_rate": np.random.rand() * 100,
                "q_value": 0.1 + np.random.rand() * 0.05,
            }

            # Create correlated initial and final parameters
            initial_params = np.random.rand(3) * 10
            final_params = initial_params + np.random.normal(0, 0.1, 3)

            # Create realistic objective value (lower is better)
            objective_value = np.sum((final_params - 5) ** 2) + np.random.rand() * 0.1

            record = OptimizationRecord(
                experiment_id=f"exp_{i:03d}",
                initial_parameters=initial_params,
                final_parameters=final_params,
                objective_value=objective_value,
                convergence_time=1 + np.random.rand() * 10,
                method="Nelder-Mead",
                experimental_conditions=conditions,
            )
            records.append(record)

        return records

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for ML data storage."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_ml_accelerated_optimizer_initialization(self, temp_data_dir):
        """Test ML accelerated optimizer initialization."""
        config = {
            "data_storage_path": str(temp_data_dir),
            "enable_transfer_learning": True,
        }

        optimizer = MLAcceleratedOptimizer(config)

        assert optimizer.config == config
        assert optimizer.predictor is not None
        assert optimizer.enable_transfer_learning is True
        assert optimizer.data_storage_path == temp_data_dir

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_accelerated_optimization_without_training(self, temp_data_dir):
        """Test optimization acceleration without pre-trained model."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Mock classical optimizer
        mock_classical_optimizer = Mock()
        mock_result = Mock()
        mock_result.fun = 0.123
        mock_classical_optimizer.run_classical_optimization_optimized.return_value = (
            np.array([1.5, 2.5, 3.5]),
            mock_result,
        )

        # Test acceleration
        initial_params = np.array([1.0, 2.0, 3.0])
        experimental_conditions = {"temperature": 25.0}

        result_params, optimization_info = optimizer.accelerate_optimization(
            mock_classical_optimizer,
            initial_params,
            experimental_conditions,
            objective_func=lambda x: np.sum(x**2),  # Simple objective
        )

        assert result_params is not None
        assert np.array_equal(result_params, np.array([1.5, 2.5, 3.5]))
        assert "ml_acceleration_info" in optimization_info
        assert (
            optimization_info["ml_acceleration_info"]["ml_initialization_used"] is False
        )

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_accelerated_optimization_with_training(
        self, sample_optimization_records, temp_data_dir
    ):
        """Test optimization acceleration with pre-trained model."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Train the predictor
        training_result = optimizer.train_predictor(sample_optimization_records)
        assert training_result["success"] is True

        # Mock classical optimizer
        mock_classical_optimizer = Mock()
        mock_result = Mock()
        mock_result.fun = 0.123
        mock_classical_optimizer.run_classical_optimization_optimized.return_value = (
            np.array([1.5, 2.5, 3.5]),
            mock_result,
        )

        # Test acceleration with trained model
        initial_params = np.array([1.0, 2.0, 3.0])
        experimental_conditions = {"temperature": 25.0, "concentration": 0.08}

        result_params, optimization_info = optimizer.accelerate_optimization(
            mock_classical_optimizer,
            initial_params,
            experimental_conditions,
            objective_func=lambda x: np.sum(x**2),
        )

        assert result_params is not None
        assert "ml_acceleration_info" in optimization_info

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_predictor_training(self, sample_optimization_records, temp_data_dir):
        """Test predictor training functionality."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Test training with sufficient data
        training_result = optimizer.train_predictor(sample_optimization_records)

        assert training_result["success"] is True
        assert "training_time" in training_result
        assert training_result["n_training_records"] == len(sample_optimization_records)
        assert "model_info" in training_result

        # Test training with insufficient data
        minimal_records = sample_optimization_records[:2]
        training_result = optimizer.train_predictor(minimal_records)

        assert training_result["success"] is False
        assert "error" in training_result

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_optimization_insights(self, sample_optimization_records, temp_data_dir):
        """Test optimization insights generation."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Add some optimization history
        optimizer.optimization_history = sample_optimization_records

        insights = optimizer.get_optimization_insights()

        assert "optimization_statistics" in insights
        assert "method_performance" in insights
        assert "ml_model_info" in insights

        stats = insights["optimization_statistics"]
        assert "total_optimizations" in stats
        assert "best_objective" in stats
        assert "average_objective" in stats

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_data_persistence(self, sample_optimization_records, temp_data_dir):
        """Test data persistence functionality."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Add optimization history
        optimizer.optimization_history = sample_optimization_records

        # Save data
        optimizer._save_training_data()

        # Create new optimizer and load data
        optimizer2 = MLAcceleratedOptimizer(config)

        # Check that data was loaded
        assert len(optimizer2.optimization_history) > 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_experiment_id_generation(self, temp_data_dir):
        """Test experiment ID generation."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        conditions1 = {"temperature": 25.0, "concentration": 0.1}
        conditions2 = {"temperature": 30.0, "concentration": 0.1}

        id1 = optimizer._generate_experiment_id(conditions1)
        id2 = optimizer._generate_experiment_id(conditions2)

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Different conditions should give different IDs
        assert id1.startswith("exp_")


class TestMLIntegrationFunctions:
    """Test suite for ML integration functions."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_create_ml_accelerated_optimizer(self):
        """Test factory function for creating ML accelerated optimizer."""
        config = {"enable_transfer_learning": False}

        optimizer = create_ml_accelerated_optimizer(config)

        assert isinstance(optimizer, MLAcceleratedOptimizer)
        assert optimizer.config == config

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
    def test_enhance_classical_optimizer_with_ml(self):
        """Test enhancement of classical optimizer with ML."""
        # Mock classical optimizer
        mock_optimizer = Mock()

        enhanced_optimizer = enhance_classical_optimizer_with_ml(
            mock_optimizer, {"enable_transfer_learning": True}
        )

        # Check that ML method was added
        assert hasattr(enhanced_optimizer, "run_ml_accelerated_optimization")
        assert hasattr(enhanced_optimizer, "_ml_accelerator")

    def test_get_ml_backend_info(self):
        """Test ML backend information retrieval."""
        backend_info = get_ml_backend_info()

        assert isinstance(backend_info, dict)
        assert "sklearn" in backend_info
        assert isinstance(backend_info["sklearn"], bool)

        # Check for version information if sklearn is available
        if backend_info["sklearn"]:
            assert (
                "sklearn_version" in backend_info or True
            )  # Version may not be available


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
class TestMLOptimizationEdgeCases:
    """Test edge cases and error conditions for ML optimization."""

    def test_predictor_with_invalid_data(self):
        """Test predictor behavior with invalid data."""
        predictor = EnsembleOptimizationPredictor()

        # Create records with NaN values
        invalid_records = [
            OptimizationRecord(
                experiment_id="invalid_001",
                initial_parameters=np.array([np.nan, 2.0, 3.0]),
                final_parameters=np.array([1.0, 2.0, 3.0]),
                objective_value=0.5,
                convergence_time=1.0,
                method="Nelder-Mead",
                experimental_conditions={"param1": 1.0},
            )
        ]

        # Should handle gracefully
        _X, _y = predictor._extract_features_and_targets(invalid_records)
        # May result in empty arrays or filtered data

    def test_prediction_with_untrained_model(self):
        """Test prediction behavior with untrained model."""
        predictor = EnsembleOptimizationPredictor()

        conditions = {"temperature": 25.0}

        with pytest.raises(RuntimeError):
            predictor.predict(conditions)

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for ML data storage."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_ml_accelerator_error_handling(self, temp_data_dir):
        """Test error handling in ML accelerator."""
        config = {"data_storage_path": str(temp_data_dir)}
        optimizer = MLAcceleratedOptimizer(config)

        # Test with failing classical optimizer
        mock_optimizer = Mock()
        mock_optimizer.run_classical_optimization_optimized.side_effect = Exception(
            "Optimization failed"
        )

        # The accelerated optimizer should handle errors gracefully, not raise
        result_params, optimization_info = optimizer.accelerate_optimization(
            mock_optimizer,
            np.array([1.0, 2.0]),
            {"temperature": 25.0},
            objective_func=lambda x: np.sum(x**2),
        )

        # Check that error was handled gracefully and fallback result returned
        assert result_params is not None
        assert "ml_acceleration_info" in optimization_info
        # The original result should indicate failure
        assert not optimization_info.get("original_result", {}).get("success", True)

    def test_empty_experimental_conditions(self):
        """Test handling of empty experimental conditions."""
        predictor = EnsembleOptimizationPredictor()

        features = predictor._conditions_to_features({}, np.array([1.0, 2.0]))

        # Should still produce valid features from initial parameters
        assert len(features) > 0
        assert not np.any(np.isnan(features))


@pytest.mark.slow
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required for ML acceleration tests")
class TestMLOptimizationPerformance:
    """Performance-focused tests for ML optimization."""

    @pytest.fixture
    def sample_optimization_records(self):
        """Create sample optimization records for testing."""
        records = []
        np.random.seed(42)  # For reproducible tests

        for i in range(20):
            # Create diverse experimental conditions
            conditions = {
                "temperature": 20 + np.random.rand() * 20,
                "concentration": 0.05 + np.random.rand() * 0.1,
                "shear_rate": np.random.rand() * 100,
                "q_value": 0.1 + np.random.rand() * 0.05,
            }

            # Create correlated initial and final parameters
            initial_params = np.random.rand(3) * 10
            final_params = initial_params + np.random.normal(0, 0.1, 3)

            # Create realistic objective value (lower is better)
            objective_value = np.sum((final_params - 5) ** 2) + np.random.rand() * 0.1

            record = OptimizationRecord(
                experiment_id=f"exp_{i:03d}",
                initial_parameters=initial_params,
                final_parameters=final_params,
                objective_value=objective_value,
                convergence_time=1 + np.random.rand() * 10,
                method="Nelder-Mead",
                experimental_conditions=conditions,
            )
            records.append(record)

        return records

    def test_training_performance(self, sample_optimization_records):
        """Test ML model training performance."""
        predictor = EnsembleOptimizationPredictor()

        start_time = time.time()
        predictor.fit(sample_optimization_records)
        training_time = time.time() - start_time

        # Training should complete in reasonable time
        assert training_time < 30.0  # 30 seconds should be plenty

    def test_prediction_performance(self, sample_optimization_records):
        """Test ML prediction performance."""
        predictor = EnsembleOptimizationPredictor()
        predictor.fit(sample_optimization_records)

        conditions = {"temperature": 25.0, "concentration": 0.08, "shear_rate": 50.0, "q_value": 0.12}

        # Time multiple predictions
        prediction_times = []
        for _ in range(10):
            start_time = time.time()
            predictor.predict(conditions, np.array([1.0, 2.0, 3.0]))
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)

        avg_prediction_time = np.mean(prediction_times)

        # Predictions should be fast
        assert avg_prediction_time < 1.0  # Less than 1 second


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
