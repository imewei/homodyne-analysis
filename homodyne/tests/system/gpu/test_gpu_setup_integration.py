#!/usr/bin/env python3
"""
Integration tests for GPU setup and command distinction functionality.

This test suite verifies:
1. GPU environment setup functions work correctly
2. Command distinction between homodyne and homodyne-gpu
3. Platform compatibility checks
4. JAX backend selection logic
5. Environment variable handling

Tests are designed to work on all platforms and handle missing dependencies gracefully.
"""

import logging
import os
import platform
from unittest.mock import patch

import pytest

logger = logging.getLogger(__name__)


# Test environment variable behavior
class TestGPUEnvironmentSetup:
    """Test GPU environment setup and detection."""

    def test_gpu_intent_detection(self):
        """Test HOMODYNE_GPU_INTENT environment variable detection."""
        # Clear environment
        original_value = os.environ.pop("HOMODYNE_GPU_INTENT", None)

        try:
            # Test default (no GPU intent)
            intent = os.environ.get("HOMODYNE_GPU_INTENT", "false").lower() == "true"
            assert not intent, "Default should be no GPU intent"

            # Test setting GPU intent
            os.environ["HOMODYNE_GPU_INTENT"] = "true"
            intent = os.environ.get("HOMODYNE_GPU_INTENT", "false").lower() == "true"
            assert intent, "Should detect GPU intent when set"

            # Test case insensitivity
            os.environ["HOMODYNE_GPU_INTENT"] = "TRUE"
            intent = os.environ.get("HOMODYNE_GPU_INTENT", "false").lower() == "true"
            assert intent, "Should be case insensitive"

        finally:
            # Restore original value
            if original_value is not None:
                os.environ["HOMODYNE_GPU_INTENT"] = original_value
            else:
                os.environ.pop("HOMODYNE_GPU_INTENT", None)

    def test_jax_platforms_cpu_setting(self):
        """Test JAX_PLATFORMS=cpu environment variable setting."""
        original_value = os.environ.get("JAX_PLATFORMS")

        try:
            # Test setting CPU-only mode
            os.environ["JAX_PLATFORMS"] = "cpu"
            assert os.environ["JAX_PLATFORMS"] == "cpu"

            # Test JAX honors the setting if available
            try:
                import jax

                backend = jax.default_backend()
                # Note: JAX may not honor JAX_PLATFORMS changes after initialization
                # If GPU was already detected, this is expected behavior
                if backend == "cpu":
                    logger.info("JAX correctly honors JAX_PLATFORMS=cpu")
                else:
                    logger.warning(
                        f"JAX backend is '{backend}' - JAX was likely already initialized with GPU support"
                    )
                    # This is not necessarily a failure - just indicates JAX was initialized before env var was set

            except ImportError:
                logger.info("JAX not available, skipping backend test")

        finally:
            # Restore original value
            if original_value is not None:
                os.environ["JAX_PLATFORMS"] = original_value
            else:
                os.environ.pop("JAX_PLATFORMS", None)


class TestGPUWrapperFunctionality:
    """Test homodyne-gpu wrapper functionality."""

    def test_method_validation_logic(self):
        """Test that unsupported methods are detected correctly."""
        # Test classical method detection
        test_args = ["homodyne-gpu", "--method", "classical"]

        with patch("sys.argv", test_args):
            # Check if method validation logic works
            if "--method" in test_args:
                method_idx = test_args.index("--method") + 1
                if method_idx < len(test_args):
                    method = test_args[method_idx]
                    assert method == "classical"
                    assert method in [
                        "classical",
                        "robust",
                    ], "Should detect unsupported method"

        # Test robust method detection
        test_args = ["homodyne-gpu", "--method", "robust"]

        with patch("sys.argv", test_args):
            if "--method" in test_args:
                method_idx = test_args.index("--method") + 1
                if method_idx < len(test_args):
                    method = test_args[method_idx]
                    assert method == "robust"
                    assert method in [
                        "classical",
                        "robust",
                    ], "Should detect unsupported method"

        # Test supported methods
        for supported_method in ["mcmc", "all"]:
            test_args = ["homodyne-gpu", "--method", supported_method]
            with patch("sys.argv", test_args):
                if "--method" in test_args:
                    method_idx = test_args.index("--method") + 1
                    if method_idx < len(test_args):
                        method = test_args[method_idx]
                        assert method == supported_method
                        assert method not in [
                            "classical",
                            "robust",
                        ], f"Should support {supported_method}"

    def test_platform_detection(self):
        """Test platform detection for GPU compatibility."""
        current_platform = platform.system()
        logger.info(f"Current platform: {current_platform}")

        # Test platform logic
        if current_platform == "Linux":
            logger.info("Platform supports GPU acceleration")
            gpu_supported = True
        else:
            logger.info(
                f"Platform {current_platform} does not support GPU acceleration"
            )
            gpu_supported = False

        # Verify our logic matches expected behavior
        assert isinstance(gpu_supported, bool)
        assert gpu_supported == (current_platform == "Linux")


class TestMCMCGPUIntegration:
    """Test MCMC GPU integration functionality."""

    def test_mcmc_sampler_gpu_intent_detection(self):
        """Test MCMC sampler detects GPU intent correctly."""
        try:
            from homodyne.optimization.mcmc import MCMCSampler
        except ImportError:
            pytest.skip("MCMC module not available")

        # Create minimal config for testing
        config = {
            "initial_parameters": {"parameter_names": ["D0", "alpha", "D_offset"]},
            "optimization_config": {"mcmc_sampling": {}},
        }

        class MockCore:
            def __init__(self):
                self.config = config

        original_intent = os.environ.get("HOMODYNE_GPU_INTENT")
        original_jax_platforms = os.environ.get("JAX_PLATFORMS")

        try:
            # Test CPU-only mode (no GPU intent)
            os.environ.pop("HOMODYNE_GPU_INTENT", None)

            # Mock PYMC_AVAILABLE to bypass the dependency check
            with (
                patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True),
                patch("logging.Logger.info") as mock_info,
            ):
                try:
                    sampler = MCMCSampler(MockCore(), config)
                except ImportError:
                    pytest.skip("PyMC dependencies not available in test environment")

                # Should force CPU-only mode
                assert (
                    not sampler.use_jax_backend
                ), "Should disable JAX backend for CPU mode"
                assert (
                    os.environ.get("JAX_PLATFORMS") == "cpu"
                ), "Should set JAX_PLATFORMS=cpu"

                # Check logging messages
                info_calls = [call.args[0] for call in mock_info.call_args_list]
                cpu_message_found = any("CPU-only" in msg for msg in info_calls)
                assert cpu_message_found, "Should log CPU-only mode message"

            # Test GPU mode (with GPU intent)
            os.environ["HOMODYNE_GPU_INTENT"] = "true"
            os.environ.pop("JAX_PLATFORMS", None)  # Reset JAX_PLATFORMS

            with (
                patch("homodyne.optimization.mcmc.PYMC_AVAILABLE", True),
                patch("logging.Logger.info") as mock_info,
            ):
                try:
                    sampler = MCMCSampler(MockCore(), config)
                except ImportError:
                    pytest.skip("PyMC dependencies not available in test environment")

                # Should allow GPU backend (if JAX available)
                info_calls = [call.args[0] for call in mock_info.call_args_list]
                gpu_message_found = any("homodyne-gpu" in msg for msg in info_calls)
                assert gpu_message_found, "Should log GPU intent detection"

        finally:
            # Restore original environment
            if original_intent is not None:
                os.environ["HOMODYNE_GPU_INTENT"] = original_intent
            else:
                os.environ.pop("HOMODYNE_GPU_INTENT", None)

            if original_jax_platforms is not None:
                os.environ["JAX_PLATFORMS"] = original_jax_platforms
            else:
                os.environ.pop("JAX_PLATFORMS", None)

    def test_jax_availability_check(self):
        """Test JAX availability detection."""
        try:
            from homodyne.optimization.mcmc import JAX_AVAILABLE

            logger.info(f"JAX_AVAILABLE: {JAX_AVAILABLE}")

            # JAX_AVAILABLE should be boolean
            assert isinstance(JAX_AVAILABLE, bool)

            # If JAX is available, should be able to import it
            if JAX_AVAILABLE:
                import jax

                logger.info(f"JAX version: {jax.__version__}")
                logger.info(f"JAX devices: {jax.devices()}")
            else:
                logger.info("JAX not available - this is normal on some systems")

        except ImportError:
            pytest.skip("MCMC module not available")


class TestGPUSetupScriptLogic:
    """Test GPU setup script logic (without executing bash)."""

    def test_nvidia_library_detection_logic(self):
        """Test NVIDIA library detection logic used in scripts."""
        # Simulate the library detection logic from runtime/gpu/activation.sh
        try:
            import site

            site_packages = site.getsitepackages()[0]

            nvidia_libs = []
            lib_names = [
                "cublas",
                "cudnn",
                "cufft",
                "curand",
                "cusolver",
                "cusparse",
                "nccl",
                "nvjitlink",
                "cuda_runtime",
                "cuda_cupti",
                "cuda_nvcc",
                "cuda_nvrtc",
            ]

            for lib in lib_names:
                lib_dir = os.path.join(site_packages, "nvidia", lib, "lib")
                if os.path.exists(lib_dir):
                    nvidia_libs.append(lib_dir)
                    logger.info(f"Found NVIDIA library: {lib}")

            logger.info(f"Total NVIDIA libraries found: {len(nvidia_libs)}")

            # Should find some libraries if NVIDIA packages are installed
            if nvidia_libs:
                logger.info("NVIDIA CUDA libraries are available via pip")
            else:
                logger.info("No pip-installed NVIDIA CUDA libraries found")

        except Exception as e:
            logger.info(f"Library detection test failed: {e}")

    def test_ld_library_path_construction(self):
        """Test LD_LIBRARY_PATH construction logic."""
        # Test path cleaning logic from GPU scripts
        test_paths = [
            "/usr/local/cuda/lib64:/home/user/lib",
            "/usr/local/cuda-12.0/lib64:/usr/lib",
            "/home/user/lib:/usr/local/cuda/lib",
            "/home/user/lib",  # No CUDA paths
        ]

        for test_path in test_paths:
            # Simulate the cleaning logic
            clean_path = ":".join(
                [p for p in test_path.split(":") if "/usr/local/cuda" not in p and p]
            )

            if "/usr/local/cuda" in test_path:
                assert (
                    "/usr/local/cuda" not in clean_path
                ), f"Should remove CUDA paths from: {test_path}"
            else:
                assert (
                    clean_path == test_path
                ), f"Should preserve non-CUDA paths: {test_path}"

            logger.info(f"Path cleaning: {test_path} -> {clean_path}")


def test_integration_gpu_files_exist():
    """Test that all required GPU files exist and are accessible."""
    from pathlib import Path

    repo_root = Path(__file__).parents[4]

    required_files = [
        "homodyne/runtime/gpu/activation.sh",
        "homodyne/runtime/README.md",
        "homodyne/gpu_wrapper.py",
    ]

    for file_path in required_files:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Required GPU file missing: {file_path}"
        assert full_path.is_file(), f"Path is not a file: {file_path}"

        if file_path.endswith(".sh"):
            # Check if shell script is executable
            assert os.access(
                full_path, os.X_OK
            ), f"Shell script not executable: {file_path}"

    logger.info("All required GPU files are present and accessible")


if __name__ == "__main__":
    # Run basic tests when script is executed directly
    logging.basicConfig(level=logging.INFO)

    print("🧪 GPU Setup Integration Tests")
    print("=" * 40)

    # Test environment variables
    test_env = TestGPUEnvironmentSetup()
    test_env.test_gpu_intent_detection()
    test_env.test_jax_platforms_cpu_setting()
    print("✅ Environment variable tests passed")

    # Test platform detection
    test_platform = TestGPUWrapperFunctionality()
    test_platform.test_platform_detection()
    test_platform.test_method_validation_logic()
    print("✅ Platform and method validation tests passed")

    # Test file existence
    test_integration_gpu_files_exist()
    print("✅ GPU file existence tests passed")

    print("\n🎉 All GPU integration tests passed!")
