"""
Shell Completion and Interactive CLI for Homodyne Analysis
==========================================================

Provides advanced CLI features including:
- Shell completion for bash, zsh, fish, and PowerShell
- Interactive mode with tab completion and real-time validation
- Context-aware suggestions and integrated help
- Command history and session management

Usage:
    # Enable shell completion (one-time setup)
    homodyne --install-completion bash    # For bash
    homodyne --install-completion zsh     # For zsh
    homodyne --install-completion fish    # For fish

    # Interactive mode
    homodyne interactive

    # Shell completion in regular commands
    homodyne --method <TAB>     # Shows: classical, mcmc, robust, all
    homodyne --config <TAB>     # Shows available .json files
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False
    argcomplete = None

try:
    import cmd2

    CMD2_AVAILABLE = True
except ImportError:
    CMD2_AVAILABLE = False
    cmd2 = None

# Readline is not directly used but checked for availability
try:
    import readline  # noqa: F401

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False


class HomodyneCompleter:
    """Custom completer for homodyne CLI arguments."""

    @staticmethod
    def config_files_completer(
        prefix: str, parsed_args: argparse.Namespace, **kwargs
    ) -> List[str]:
        """Complete configuration file paths."""
        # Look for .json files in current directory and common locations
        patterns = [
            "*.json",
            "config*.json",
            "homodyne*.json",
            "./configs/*.json",
            "./data/*.json",
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

        # Filter by prefix
        return [f for f in files if f.startswith(prefix)]

    @staticmethod
    def output_dir_completer(
        prefix: str, parsed_args: argparse.Namespace, **kwargs
    ) -> List[str]:
        """Complete directory paths."""
        # Get directories that match the prefix
        if not prefix:
            prefix = "./"

        try:
            base_dir = os.path.dirname(prefix) or "."
            partial_name = os.path.basename(prefix)

            dirs = []
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    if item.startswith(partial_name):
                        dirs.append(os.path.join(base_dir, item))

            return dirs
        except (OSError, FileNotFoundError):
            return []

    @staticmethod
    def method_completer(
        prefix: str, parsed_args: argparse.Namespace, **kwargs
    ) -> List[str]:
        """Complete method choices with descriptions."""
        methods = {
            "classical": "Classical optimization (Nelder-Mead, Gurobi)",
            "mcmc": "Bayesian MCMC sampling (NUTS)",
            "robust": "Robust optimization (Wasserstein, Scenario, Ellipsoidal)",
            "all": "Run all methods (Classical + Robust + MCMC)",
        }
        return [method for method in methods.keys() if method.startswith(prefix)]

    @staticmethod
    def analysis_mode_completer(
        prefix: str, parsed_args: argparse.Namespace, **kwargs
    ) -> List[str]:
        """Complete analysis mode choices."""
        modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        return [mode for mode in modes if mode.startswith(prefix)]


class InteractiveHomodyneCLI:
    """Interactive CLI for homodyne analysis with tab completion and validation."""

    # Type hints for dynamically assigned cmd2 methods
    poutput: callable  # type: ignore[misc]
    cmdloop: callable  # type: ignore[misc]
    onecmd: callable  # type: ignore[misc]
    parseline: callable  # type: ignore[misc]
    completedefault: callable  # type: ignore[misc]

    def __init__(self):
        if not CMD2_AVAILABLE:
            raise ImportError(
                "cmd2 package is required for interactive mode. "
                "Install with: pip install cmd2"
            )

        # Initialize cmd2 functionality if available
        if cmd2 is not None:
            # Create a cmd2 instance and copy its methods
            cmd_instance = cmd2.Cmd()
            # Copy essential cmd2 methods and attributes
            for attr in [
                "poutput",
                "cmdloop",
                "onecmd",
                "parseline",
                "completedefault",
                "intro",
                "prompt",
            ]:
                if hasattr(cmd_instance, attr):
                    setattr(self, attr, getattr(cmd_instance, attr))

        self.intro = """
╭─────────────────────────────────────────────────────────╮
│            Homodyne Analysis Interactive CLI            │
│                                                         │
│  Tab completion, command history, and real-time help   │
│  Type 'help' for commands or 'help <command>' for info │
│  Use Ctrl+C to exit or type 'quit'                     │
╰─────────────────────────────────────────────────────────╯
        """
        self.prompt = "homodyne> "

        # Current session state
        self.current_config = None
        self.current_method = "classical"
        self.current_output_dir = "./homodyne_results"
        self.verbose = False

        # Load available configs
        self._load_available_configs()

    def _load_available_configs(self):
        """Load available configuration files."""
        self.available_configs = []
        patterns = ["*.json", "config*.json", "homodyne*.json"]
        for pattern in patterns:
            self.available_configs.extend(glob.glob(pattern))

    def do_run(self, args):
        """Run homodyne analysis with current settings.

        Usage: run [--method METHOD] [--config CONFIG] [additional options...]

        Examples:
            run                           # Run with current settings
            run --method mcmc             # Run MCMC with current config
            run --config new_config.json  # Run with different config
        """
        # Parse additional arguments
        parser = argparse.ArgumentParser(prog="run")
        parser.add_argument(
            "--method",
            choices=["classical", "mcmc", "robust", "all"],
            default=self.current_method,
        )
        parser.add_argument("--config", default=self.current_config)
        parser.add_argument("--output-dir", default=self.current_output_dir)
        parser.add_argument("--verbose", action="store_true", default=self.verbose)
        parser.add_argument("--static-isotropic", action="store_true")
        parser.add_argument("--static-anisotropic", action="store_true")
        parser.add_argument("--laminar-flow", action="store_true")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        # Update current state
        self.current_method = parsed_args.method
        if parsed_args.config:
            self.current_config = parsed_args.config
        self.current_output_dir = parsed_args.output_dir
        self.verbose = parsed_args.verbose

        # Build command for actual execution
        cmd_parts = ["python", "-m", "homodyne.run_homodyne"]
        cmd_parts.extend(["--method", parsed_args.method])

        if parsed_args.config:
            cmd_parts.extend(["--config", parsed_args.config])

        cmd_parts.extend(["--output-dir", parsed_args.output_dir])

        if parsed_args.verbose:
            cmd_parts.append("--verbose")

        if parsed_args.static_isotropic:
            cmd_parts.append("--static-isotropic")
        elif parsed_args.static_anisotropic:
            cmd_parts.append("--static-anisotropic")
        elif parsed_args.laminar_flow:
            cmd_parts.append("--laminar-flow")

        self.poutput(f"Executing: {' '.join(cmd_parts)}")

        # Import and run the main function
        try:
            from .run_homodyne import main as run_main

            # Temporarily override sys.argv
            old_argv = sys.argv.copy()
            sys.argv = cmd_parts

            try:
                result = run_main()
                if result == 0:
                    self.poutput("✓ Analysis completed successfully!")
                else:
                    self.poutput(f"✗ Analysis failed with code {result}")
            finally:
                sys.argv = old_argv

        except Exception as e:
            self.poutput(f"Error: {e}")

    def complete_run(self, text, line, begidx, endidx):
        """Tab completion for run command."""
        if "--method" in line:
            return [
                m for m in ["classical", "mcmc", "robust", "all"] if m.startswith(text)
            ]
        elif "--config" in line:
            return [f for f in self.available_configs if f.startswith(text)]
        else:
            return [
                "--method",
                "--config",
                "--output-dir",
                "--verbose",
                "--static-isotropic",
                "--static-anisotropic",
                "--laminar-flow",
            ]

    def do_config(self, args):
        """Set or show current configuration.

        Usage:
            config                    # Show current config
            config set <file>         # Set config file
            config show               # Display config contents
            config validate           # Validate current config
        """
        if not args:
            self.poutput(f"Current config: {self.current_config or 'None'}")
            return

        parts = args.split()
        if parts[0] == "set":
            if len(parts) < 2:
                self.poutput("Usage: config set <file>")
                return

            config_file = parts[1]
            if not os.path.exists(config_file):
                self.poutput(f"Error: Config file '{config_file}' not found")
                return

            self.current_config = config_file
            self.poutput(f"✓ Config set to: {config_file}")

        elif parts[0] == "show":
            if not self.current_config:
                self.poutput("No config file set")
                return

            try:
                with open(self.current_config, "r") as f:
                    config = json.load(f)
                self.poutput(json.dumps(config, indent=2))
            except Exception as e:
                self.poutput(f"Error reading config: {e}")

        elif parts[0] == "validate":
            if not self.current_config:
                self.poutput("No config file set")
                return

            try:
                from .core.config import ConfigManager

                ConfigManager(self.current_config)  # Just validate
                self.poutput("✓ Configuration is valid")
            except Exception as e:
                self.poutput(f"✗ Configuration error: {e}")

    def complete_config(self, text, line, begidx, endidx):
        """Tab completion for config command."""
        if "set" in line:
            return [f for f in self.available_configs if f.startswith(text)]
        else:
            return [cmd for cmd in ["set", "show", "validate"] if cmd.startswith(text)]

    def do_method(self, args):
        """Set or show current analysis method.

        Usage:
            method                    # Show current method
            method <method>           # Set method (classical, mcmc, robust, all)
        """
        if not args:
            self.poutput(f"Current method: {self.current_method}")
            methods_info = {
                "classical": "Classical optimization (Nelder-Mead, Gurobi)",
                "mcmc": "Bayesian MCMC sampling (NUTS)",
                "robust": "Robust optimization (Wasserstein, Scenario, Ellipsoidal)",
                "all": "Run all methods (Classical + Robust + MCMC)",
            }
            self.poutput("\nAvailable methods:")
            for method, desc in methods_info.items():
                marker = "→" if method == self.current_method else " "
                self.poutput(f"  {marker} {method:10} - {desc}")
            return

        method = args.strip()
        if method not in ["classical", "mcmc", "robust", "all"]:
            self.poutput(
                f"Error: Invalid method '{method}'. "
                "Choose from: classical, mcmc, robust, all"
            )
            return

        self.current_method = method
        self.poutput(f"✓ Method set to: {method}")

    def complete_method(self, text, line, begidx, endidx):
        """Tab completion for method command."""
        methods = ["classical", "mcmc", "robust", "all"]
        return [m for m in methods if m.startswith(text)]

    def do_status(self, args):
        """Show current session status and settings."""
        self.poutput("╭─────────────────────────────────────────╮")
        self.poutput("│           Current Settings             │")
        self.poutput("├─────────────────────────────────────────┤")
        self.poutput(f"│ Method     : {self.current_method:<25} │")
        self.poutput(f"│ Config     : {(self.current_config or 'None'):<25} │")
        self.poutput(f"│ Output Dir : {self.current_output_dir:<25} │")
        self.poutput(f"│ Verbose    : {self.verbose:<25} │")
        self.poutput("╰─────────────────────────────────────────╯")

    def do_ls(self, args):
        """List files in current directory or specified directory."""
        import os

        directory = args.strip() or "."
        try:
            files = os.listdir(directory)
            # Separate and highlight different file types
            configs = [f for f in files if f.endswith(".json")]
            dirs = [f for f in files if os.path.isdir(os.path.join(directory, f))]
            others = [
                f
                for f in files
                if not f.endswith(".json")
                and not os.path.isdir(os.path.join(directory, f))
            ]

            if configs:
                self.poutput("Config files:")
                for config in sorted(configs):
                    self.poutput(f"  📄 {config}")

            if dirs:
                self.poutput("Directories:")
                for d in sorted(dirs):
                    self.poutput(f"  📁 {d}")

            if others:
                self.poutput("Other files:")
                for other in sorted(others):
                    self.poutput(f"  📎 {other}")

        except OSError as e:
            self.poutput(f"Error: {e}")

    def do_create_config(self, args):
        """Create a new configuration file interactively.

        Usage: create_config [--mode MODE] [--sample SAMPLE] [--output FILE]
        """
        parser = argparse.ArgumentParser(prog="create_config")
        parser.add_argument(
            "--mode",
            choices=["static_isotropic", "static_anisotropic", "laminar_flow"],
            default="laminar_flow",
        )
        parser.add_argument("--sample", help="Sample name")
        parser.add_argument(
            "--output", default="my_config.json", help="Output filename"
        )

        try:
            parsed_args = parser.parse_args(args.split() if args else [])
        except SystemExit:
            return

        try:
            from .create_config import create_config_from_template

            create_config_from_template(
                output_file=parsed_args.output,
                sample_name=parsed_args.sample,
                mode=parsed_args.mode,
            )

            # Update available configs
            self._load_available_configs()

            # Offer to set as current config
            response = input(f"Set {parsed_args.output} as current config? (y/n): ")
            if response.lower().startswith("y"):
                self.current_config = parsed_args.output
                self.poutput(f"✓ Current config set to: {parsed_args.output}")

        except Exception as e:
            self.poutput(f"Error creating config: {e}")

    def complete_create_config(self, text, line, begidx, endidx):
        """Tab completion for create_config command."""
        if "--mode" in line:
            modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
            return [m for m in modes if m.startswith(text)]
        else:
            return ["--mode", "--sample", "--output"]

    def do_help_guide(self, args):
        """Show comprehensive usage guide."""
        guide = """
╭─────────────────────────────────────────────────────────────────╮
│                     Homodyne Interactive Guide                  │
╰─────────────────────────────────────────────────────────────────╯

🚀 Getting Started:
   1. config set <file>     - Load a configuration file
   2. method <type>         - Choose analysis method
   3. run                   - Execute analysis

📋 Essential Commands:
   • status                 - Show current settings
   • ls                     - List files and configs
   • create_config          - Create new configuration
   • config validate        - Check config file validity

🔧 Analysis Methods:
   • classical             - Fast optimization (Nelder-Mead, Gurobi)
   • mcmc                  - Bayesian sampling for uncertainty
   • robust                - Noise-resistant optimization
   • all                   - Run all methods

📁 File Management:
   • ls                    - List directory contents
   • config show           - View configuration content
   • Tab completion        - Works for files and commands

💡 Pro Tips:
   • Use Tab for autocompletion
   • Up/Down arrows for command history
   • Ctrl+C to interrupt, 'quit' to exit
   • help <command> for specific help

Example Workflow:
   1. ls                           # See available files
   2. config set my_config.json    # Load configuration
   3. method mcmc                  # Set analysis method
   4. run --verbose                # Execute with logging
        """
        self.poutput(guide)


def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
    """Add shell completion support to argument parser."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        return

    # Add completers to specific arguments
    for action in parser._actions:  # noqa: W291
        if action.dest == "method":
            # Use setattr to avoid type checker issues with dynamic attribute
            setattr(action, "completer", HomodyneCompleter.method_completer)
        elif action.dest == "config":
            setattr(action, "completer", HomodyneCompleter.config_files_completer)
        elif action.dest == "output_dir":
            setattr(action, "completer", HomodyneCompleter.output_dir_completer)

    # Enable argcomplete
    argcomplete.autocomplete(parser)


def install_shell_completion(shell: str) -> int:
    """Install shell completion for the specified shell."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        print("Error: argcomplete package is required for shell completion.")
        print("Install with: pip install argcomplete")
        return 1

    completion_scripts = {
        "bash": """# Homodyne completion for bash
eval "$(register-python-argcomplete homodyne)"
eval "$(register-python-argcomplete homodyne-config)"
""",
        "zsh": """# Homodyne completion for zsh
eval "$(register-python-argcomplete homodyne)"
eval "$(register-python-argcomplete homodyne-config)"
""",
        "fish": """# Homodyne completion for fish
register-python-argcomplete --shell fish homodyne | source
register-python-argcomplete --shell fish homodyne-config | source
""",
        "powershell": """# Homodyne completion for PowerShell
Register-ArgumentCompleter -Native -CommandName homodyne -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $Env:_ARGCOMPLETE_COMP_WORDBREAKS = ' \t\n'
    $Env:_ARGCOMPLETE = 1
    $Env:_ARGCOMPLETE_SUPPRESS_SPACE = 1
    $Env:COMP_LINE = $commandAst
    $Env:COMP_POINT = $cursorPosition
    homodyne 2>&1 | Where-Object { $_ -like "$wordToComplete*" }
}
""",
    }

    if shell not in completion_scripts:
        supported_shells = ', '.join(completion_scripts.keys())
        print(f"Error: Shell '{shell}' not supported. Choose from: {supported_shells}")
        return 1

    script = completion_scripts[shell]

    # Determine the appropriate config file
    home = Path.home()
    config_files = {
        "bash": home / ".bashrc",
        "zsh": home / ".zshrc",
        "fish": home / ".config" / "fish" / "config.fish",
        "powershell": home
        / "Documents"
        / "PowerShell"
        / "Microsoft.PowerShell_profile.ps1",
    }

    config_file = config_files[shell]

    print(f"Installing {shell} completion for homodyne...")

    try:
        # Create directory if needed
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if completion is already installed
        if config_file.exists():
            content = config_file.read_text()
            if "homodyne" in content and "argcomplete" in content:
                print(f"✓ Completion already installed in {config_file}")
                return 0

        # Append completion script
        with open(config_file, "a", encoding="utf-8") as f:
            f.write(f"\n{script}\n")

        print(f"✓ Completion installed in {config_file}")
        print(f"✓ Restart your {shell} session or run: source {config_file}")
        print("\nCompletion features:")
        print("  • Tab completion for --method, --config, --output-dir")
        print("  • File path completion for configuration files")
        print("  • Context-aware suggestions")

        return 0

    except Exception as e:
        print(f"Error installing completion: {e}")
        return 1


def start_interactive_mode() -> int:
    """Start interactive CLI mode."""
    try:
        cli = InteractiveHomodyneCLI()
        cli.cmdloop()
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error in interactive mode: {e}")
        return 1


# Export public functions
__all__ = [
    "setup_shell_completion",
    "install_shell_completion",
    "start_interactive_mode",
    "HomodyneCompleter",
    "InteractiveHomodyneCLI",
]
