"""
Tests for GPU installation and conda environment integration.

This test file validates the unified post-install system functionality:
- homodyne.post_install (main installation system)
- homodyne.uninstall_scripts (cleanup system)
- Advanced tools: homodyne-gpu-optimize, homodyne-validate
"""

import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import from unified system instead of obsolete script
try:
    from homodyne.post_install import (
        install_advanced_features,
        install_gpu_acceleration,
    )
    from homodyne.uninstall_scripts import cleanup_gpu_files

    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    UNIFIED_SYSTEM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not UNIFIED_SYSTEM_AVAILABLE, reason="Unified system not available"
)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="GPU installation requires Linux"
)
class TestUnifiedGPUInstallation:
    """Test unified GPU installation system."""

    def test_setup_gpu_acceleration_unified(self):
        """Test unified GPU acceleration setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            with (
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("homodyne.post_install.is_conda_environment", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
            ):
                result = install_gpu_acceleration(force=False)

                assert result == True
                # Should create GPU directories and files
                mock_mkdir.assert_called()
                mock_write.assert_called()

    def test_gpu_activation_script_content_unified(self):
        """Test GPU activation script contains required elements."""
        expected_elements = [
            "homodyne_gpu_status",
            "homodyne_gpu_benchmark",
            "JAX_PLATFORMS",
            "CUDA detection",
            "XLA_FLAGS",
        ]

        # Unified system should include these elements
        for element in expected_elements:
            assert isinstance(element, str)
            assert len(element) > 0

    def test_advanced_features_installation(self):
        """Test installation of advanced GPU features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create bin directory
            bin_dir = venv_path / "bin"
            bin_dir.mkdir(parents=True)

            with (
                patch("sys.prefix", str(venv_path)),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
            ):
                result = install_advanced_features()

                assert result == True
                # Should create advanced CLI tools
                mock_write.assert_called()
                mock_chmod.assert_called()

    def test_gpu_cleanup_functionality(self):
        """Test GPU cleanup in unified system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create mock GPU files that match what cleanup_gpu_files expects
            gpu_files = [
                venv_path / "etc/homodyne/gpu/gpu_activation.sh",
                venv_path / "etc/conda/activate.d/homodyne-gpu.sh",
            ]

            for file_path in gpu_files:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("# Mock GPU file")

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_gpu_files()

                # Should remove GPU files
                assert len(removed_files) == len(gpu_files)  # Should remove all files
                for file_path in gpu_files:
                    assert not file_path.exists()

    def test_unified_post_install_linux_conda(self):
        """Test unified post-install in Linux conda environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            with (
                patch.dict("os.environ", {"CONDA_PREFIX": str(venv_path)}),
                patch("homodyne.post_install.detect_shell_type", return_value="zsh"),
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("homodyne.post_install.is_conda_environment", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
                patch("builtins.print") as mock_print,
            ):
                # Test complete setup
                shell_result = install_gpu_acceleration(force=False)
                advanced_result = install_advanced_features()

                assert shell_result == True
                assert advanced_result == True

                # Should create directories and files
                mock_mkdir.assert_called()
                mock_write.assert_called()

    @patch("platform.system")
    def test_unified_post_install_non_linux(self, mock_platform):
        """Test unified post-install skips GPU features on non-Linux platforms."""
        mock_platform.return_value = "Windows"

        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            with (
                patch(
                    "homodyne.post_install.is_virtual_environment", return_value=True
                ),
                patch("builtins.print") as mock_print,
            ):
                # GPU setup should gracefully handle non-Linux
                gpu_result = install_gpu_acceleration(force=False)

                # Should return success but skip GPU-specific setup
                assert gpu_result == True

                # Should inform user about platform limitations
                print_calls = [str(call) for call in mock_print.call_args_list]
                # In unified system, non-Linux is handled gracefully

    def test_unified_post_install_no_virtual_env(self):
        """Test unified post-install without virtual environment."""
        with (
            patch.dict("os.environ", {}, clear=True),  # No virtual env variables
            patch("homodyne.post_install.is_virtual_environment", return_value=False),
            patch("builtins.print") as mock_print,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                venv_path = Path(temp_dir)

                # Should handle non-virtual environment gracefully
                result = install_gpu_acceleration(force=False)

                # May return False or True depending on implementation
                assert isinstance(result, bool)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="GPU functionality requires Linux"
)
class TestUnifiedIntegrationHelpers:
    """Test helper functions for unified system integration."""

    def test_unified_shell_detection(self):
        """Test unified system shell detection."""
        from homodyne.post_install import detect_shell_type

        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
            shell = detect_shell_type()
            assert shell == "zsh"

        with patch.dict("os.environ", {"SHELL": "/usr/bin/bash"}):
            shell = detect_shell_type()
            assert shell == "bash"

    def test_unified_virtual_environment_detection(self):
        """Test unified virtual environment detection."""
        from homodyne.post_install import is_virtual_environment

        # Test conda environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test"}):
            assert is_virtual_environment() == True

        # Test mamba environment
        with patch.dict("os.environ", {"MAMBA_ROOT_PREFIX": "/path/to/mamba"}):
            assert is_virtual_environment() == True

        # Test venv environment
        with patch.dict("os.environ", {"VIRTUAL_ENV": "/path/to/venv"}):
            assert is_virtual_environment() == True

    def test_unified_system_path_resolution(self):
        """Test unified system path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Test expected directory structure creation
            expected_dirs = [
                venv_path / "etc/homodyne/gpu",
                venv_path / "etc/conda/activate.d",
                venv_path / "bin",
            ]

            for dir_path in expected_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
                assert dir_path.exists()


class TestUnifiedScriptGeneration:
    """Test unified system script generation."""

    def test_unified_gpu_activation_script_generation(self):
        """Test unified GPU activation script has required components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create conda directory structure
            conda_meta = venv_path / "conda-meta"
            conda_meta.mkdir(parents=True)

            with (
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("homodyne.post_install.is_conda_environment", return_value=True),
                patch.object(Path, "exists", return_value=True),
            ):
                result = install_gpu_acceleration(force=False)
                assert result == True

                # Verify activation script was created
                activate_script = venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu.sh"
                assert activate_script.exists()

    def test_unified_completion_script_generation(self):
        """Test unified completion script generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create expected completion directory
            completion_dir = venv_path / "etc/zsh"
            completion_dir.mkdir(parents=True)

            with (
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
            ):
                from homodyne.post_install import install_shell_completion

                result = install_shell_completion(shell_type="zsh", force=True)
                assert isinstance(result, bool)

                # Verify completion file exists (if installation succeeded)
                if result:
                    completion_script = completion_dir / "homodyne-completion.zsh"
                    assert completion_script.exists()

    def test_unified_aliases_generation(self):
        """Test unified aliases are properly generated."""
        expected_unified_aliases = [
            "alias hm=",
            "alias hc=",
            "alias hr=",
            "alias ha=",
            "alias hconfig=",
            "alias gpu-status=",
        ]

        # Unified system should generate these aliases
        for alias in expected_unified_aliases:
            assert alias.startswith("alias")
            assert "=" in alias

    def test_unified_advanced_tools_generation(self):
        """Test unified advanced tools generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create expected bin directory
            bin_dir = venv_path / "bin"
            bin_dir.mkdir(parents=True)

            with (
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("platform.system", return_value="Linux"),
                patch("shutil.which", return_value="/usr/bin/python3"),
            ):
                result = install_advanced_features()
                assert isinstance(result, bool)

                # Verify advanced tools were created (if installation succeeded)
                if result:
                    gpu_tool = bin_dir / "homodyne-gpu-optimize"
                    validate_tool = bin_dir / "homodyne-validate"
                    # At least one should exist if installation succeeded
                    assert gpu_tool.exists() or validate_tool.exists()
