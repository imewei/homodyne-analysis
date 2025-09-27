"""
Configuration templates and utilities for homodyne analysis.

This module provides utilities for accessing configuration templates moved from the root directory.
"""

from pathlib import Path

# Configuration template directory
CONFIG_DIR = Path(__file__).parent

# Template file mapping for backward compatibility
TEMPLATE_FILES = {
    "static_isotropic": "static_isotropic.json",
    "static_anisotropic": "static_anisotropic.json",
    "laminar_flow": "laminar_flow.json",
    "template": "template.json",
}


def get_template_path(template_name: str) -> Path:
    """
    Get the path to a configuration template file.

    Parameters
    ----------
    template_name : str
        Name of the template ('static_isotropic', 'static_anisotropic', 'laminar_flow', 'template')

    Returns
    -------
    Path
        Path to the template file
    """
    if template_name not in TEMPLATE_FILES:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(TEMPLATE_FILES.keys())}"
        )

    return CONFIG_DIR / TEMPLATE_FILES[template_name]


def get_config_dir() -> Path:
    """
    Get the configuration directory path.

    Returns
    -------
    Path
        Path to the configuration directory
    """
    return CONFIG_DIR


__all__ = [
    "CONFIG_DIR",
    "TEMPLATE_FILES",
    "get_config_dir",
    "get_template_path",
]
