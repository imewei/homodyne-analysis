"""
Fast Completion Handler for CLI
===============================

Ultra-fast completion handler that bypasses heavy imports for immediate
argcomplete response. This integrates with the advanced completion system
while maintaining optimal performance.
"""

import os
import sys
from pathlib import Path


def handle_fast_completion():
    """
    Ultra-fast completion handler that bypasses all heavy imports.

    This function is called very early in the CLI startup process to handle
    argcomplete requests without loading the full application.
    """
    # Only handle completion if argcomplete is actively requesting completions
    if os.environ.get("_ARGCOMPLETE") != "1":
        return False

    try:
        # Get completion context from environment
        comp_line = os.environ.get("COMP_LINE", "")
        comp_point = int(os.environ.get("COMP_POINT", len(comp_line)))

        # Parse command line up to cursor position
        words = comp_line[:comp_point].split()

        if len(words) >= 1:
            # Determine context
            if comp_line[comp_point - 1 : comp_point].isspace():
                # Space after last word - completing the value for that argument
                prev_word = words[-1] if words else ""
                current_word = ""
            else:
                # No space - still typing the current word
                prev_word = words[-2] if len(words) >= 2 else ""
                current_word = words[-1] if words else ""

            # Fast completion based on previous word
            completions = []

            if prev_word in ["--method", "-m"]:
                methods = ["vi", "mcmc", "hybrid"]  # Updated for new system
                if current_word:
                    methods = [m for m in methods if m.startswith(current_word)]
                completions = methods

            elif prev_word in ["--config", "-c"]:
                # Fast config file completion
                try:
                    cwd = Path.cwd()
                    json_files = [
                        f.name
                        for f in cwd.iterdir()
                        if f.is_file() and f.suffix == ".json"
                    ]
                    # Prioritize common config files
                    priority = [
                        "config.json",
                        "homodyne_config.json",
                        "analysis_config.json",
                        "my_config.json",
                    ]
                    result = [f for f in priority if f in json_files]
                    result.extend([f for f in json_files if f not in priority][:8])

                    if current_word:
                        result = [
                            f
                            for f in result
                            if f.lower().startswith(current_word.lower())
                        ]
                    completions = result
                except Exception:
                    # Fallback to common config files if directory scan fails
                    completions = ["config.json", "homodyne_config.json"]

            elif prev_word in ["--output-dir", "-o"]:
                # Fast directory completion
                try:
                    cwd = Path.cwd()
                    dirs = [d.name for d in cwd.iterdir() if d.is_dir()]
                    # Prioritize common output directories
                    priority = ["output", "results", "data", "plots", "analysis"]
                    result = [d for d in priority if d in dirs]
                    result.extend([d for d in dirs if d not in priority][:5])

                    if current_word:
                        result = [
                            d
                            for d in result
                            if d.lower().startswith(current_word.lower())
                        ]
                    completions = [d + "/" for d in result]
                except Exception:
                    # Fallback to common directories
                    completions = ["output/", "results/", "data/"]

            elif prev_word in ["--mode"]:
                modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
                if current_word:
                    modes = [m for m in modes if m.startswith(current_word)]
                completions = modes

            elif prev_word in ["--install-completion", "--uninstall-completion"]:
                shells = ["bash", "zsh", "fish"]
                if current_word:
                    shells = [s for s in shells if s.startswith(current_word)]
                completions = shells

            elif current_word.startswith("--"):
                # Complete option flags
                options = [
                    "--method",
                    "--config",
                    "--output-dir",
                    "--verbose",
                    "--quiet",
                    "--static-isotropic",
                    "--static-anisotropic",
                    "--laminar-flow",
                    "--plot-experimental-data",
                    "--plot-simulated-data",
                    "--install-completion",
                    "--uninstall-completion",
                    "--help",
                ]
                completions = [opt for opt in options if opt.startswith(current_word)]

            # Output completions and exit
            for completion in completions:
                print(completion)
            sys.exit(0)

    except Exception:
        # If fast completion fails, let the normal system handle it
        pass

    return False
