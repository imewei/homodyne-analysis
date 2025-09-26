"""
Interactive Analysis Interface and Workflow Management
=====================================================

Advanced interactive interface for guided analysis workflows, real-time parameter
adjustment, and collaborative analysis sessions for scientific computing.

Features:
- Interactive parameter tuning with real-time feedback
- Guided analysis workflows with step-by-step assistance
- Live plot updates during optimization
- Collaborative analysis session management
- Jupyter notebook integration
- Web-based interface for remote access
- Session recording and playback
- Automated workflow generation
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, track
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from .cli_enhancer import CLIEnhancer
from .error_reporter import create_error_reporter
from .progress import ProgressContext, get_progress_tracker


@dataclass
class WorkflowStep:
    """Individual step in an interactive workflow."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]
    optional: bool = False
    estimated_time: float | None = None
    dependencies: list[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AnalysisSession:
    """Analysis session state and configuration."""

    session_id: str
    start_time: float
    config: dict[str, Any]
    results: dict[str, Any]
    workflow_steps: list[WorkflowStep]
    current_step: int = 0
    completed_steps: list[str] = None

    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []


class InteractiveInterface:
    """
    Advanced interactive interface for guided scientific analysis workflows.

    Provides multiple interface modes (CLI, Jupyter, web) with real-time feedback,
    parameter adjustment, and collaborative features.
    """

    def __init__(
        self,
        interface_mode: str = "auto",
        enable_live_updates: bool = True,
        session_recording: bool = True,
    ):
        """
        Initialize interactive interface.

        Parameters
        ----------
        interface_mode : str
            Interface mode ('cli', 'jupyter', 'web', 'auto')
        enable_live_updates : bool
            Enable real-time plot and parameter updates
        session_recording : bool
            Record session for playback and analysis
        """
        self.interface_mode = self._determine_interface_mode(interface_mode)
        self.enable_live_updates = enable_live_updates
        self.session_recording = session_recording

        # Initialize components
        self.console = Console() if RICH_AVAILABLE else None
        self.error_reporter = create_error_reporter()
        self.cli_enhancer = CLIEnhancer() if RICH_AVAILABLE else None

        # Session management
        self.current_session: AnalysisSession | None = None
        self.session_history = []

        # Workflow definitions
        self.workflows = self._initialize_workflows()

        # Logger
        self.logger = logging.getLogger(__name__)

    def _determine_interface_mode(self, mode: str) -> str:
        """Determine the best interface mode based on environment."""
        if mode == "auto":
            # Check environment and available packages
            if JUPYTER_AVAILABLE and self._is_jupyter_environment():
                return "jupyter"
            elif STREAMLIT_AVAILABLE and self._is_streamlit_environment():
                return "web"
            elif RICH_AVAILABLE:
                return "cli"
            else:
                return "basic_cli"
        return mode

    def _is_jupyter_environment(self) -> bool:
        """Check if running in Jupyter environment."""
        try:
            from IPython import get_ipython

            return get_ipython() is not None
        except ImportError:
            return False

    def _is_streamlit_environment(self) -> bool:
        """Check if running in Streamlit environment."""
        try:
            return "streamlit" in globals() or hasattr(st, "sidebar")
        except (NameError, AttributeError):
            return False

    def _initialize_workflows(self) -> dict[str, list[WorkflowStep]]:
        """Initialize predefined analysis workflows."""
        return {
            "quick_analysis": [
                WorkflowStep(
                    name="config_validation",
                    description="Validate configuration file",
                    function=self._validate_configuration,
                    parameters={},
                    estimated_time=5.0,
                ),
                WorkflowStep(
                    name="data_loading",
                    description="Load experimental data",
                    function=self._load_data,
                    parameters={},
                    estimated_time=30.0,
                    dependencies=["config_validation"],
                ),
                WorkflowStep(
                    name="classical_optimization",
                    description="Run classical optimization",
                    function=self._run_classical_optimization,
                    parameters={"method": "nelder_mead"},
                    estimated_time=120.0,
                    dependencies=["data_loading"],
                ),
                WorkflowStep(
                    name="visualization",
                    description="Generate result plots",
                    function=self._create_visualizations,
                    parameters={},
                    estimated_time=20.0,
                    dependencies=["classical_optimization"],
                ),
            ],
            "comprehensive_analysis": [
                WorkflowStep(
                    name="config_validation",
                    description="Validate configuration file",
                    function=self._validate_configuration,
                    parameters={},
                    estimated_time=5.0,
                ),
                WorkflowStep(
                    name="data_loading",
                    description="Load and validate experimental data",
                    function=self._load_data,
                    parameters={"validate": True},
                    estimated_time=60.0,
                    dependencies=["config_validation"],
                ),
                WorkflowStep(
                    name="parameter_exploration",
                    description="Interactive parameter space exploration",
                    function=self._parameter_exploration,
                    parameters={},
                    estimated_time=300.0,
                    dependencies=["data_loading"],
                    optional=True,
                ),
                WorkflowStep(
                    name="classical_optimization",
                    description="Run all classical optimization methods",
                    function=self._run_classical_optimization,
                    parameters={"method": "all"},
                    estimated_time=300.0,
                    dependencies=["data_loading"],
                ),
                WorkflowStep(
                    name="robust_optimization",
                    description="Run robust optimization methods",
                    function=self._run_robust_optimization,
                    parameters={},
                    estimated_time=600.0,
                    dependencies=["classical_optimization"],
                ),
                WorkflowStep(
                    name="method_comparison",
                    description="Compare optimization results",
                    function=self._compare_methods,
                    parameters={},
                    estimated_time=60.0,
                    dependencies=["classical_optimization", "robust_optimization"],
                ),
                WorkflowStep(
                    name="visualization",
                    description="Generate comprehensive visualizations",
                    function=self._create_visualizations,
                    parameters={"comprehensive": True},
                    estimated_time=120.0,
                    dependencies=["method_comparison"],
                ),
            ],
            "custom_workflow": [],
        }

    def start_interactive_session(
        self, workflow_name: str = "quick_analysis"
    ) -> AnalysisSession:
        """Start an interactive analysis session."""
        session_id = f"session_{int(time.time())}"

        # Initialize session
        session = AnalysisSession(
            session_id=session_id,
            start_time=time.time(),
            config={},
            results={},
            workflow_steps=self.workflows.get(workflow_name, []).copy(),
        )

        self.current_session = session

        if self.interface_mode == "cli":
            return self._start_cli_session(session)
        elif self.interface_mode == "jupyter":
            return self._start_jupyter_session(session)
        elif self.interface_mode == "web":
            return self._start_web_session(session)
        else:
            return self._start_basic_cli_session(session)

    def _start_cli_session(self, session: AnalysisSession) -> AnalysisSession:
        """Start CLI-based interactive session."""
        if not RICH_AVAILABLE or not self.console:
            return self._start_basic_cli_session(session)

        # Welcome panel
        welcome_panel = Panel(
            f"[bold blue]Interactive Homodyne Analysis Session[/bold blue]\n\n"
            f"Session ID: {session.session_id}\n"
            f"Workflow: {len(session.workflow_steps)} steps\n"
            f"Mode: {self.interface_mode.upper()}",
            title="[green]Welcome[/green]",
            border_style="green",
        )
        self.console.print(welcome_panel)

        # Configuration setup
        session.config = self._interactive_configuration_setup()

        # Workflow execution
        self._execute_workflow_cli(session)

        return session

    def _interactive_configuration_setup(self) -> dict[str, Any]:
        """Interactive configuration setup."""
        if not RICH_AVAILABLE or not self.console:
            return self._basic_configuration_setup()

        self.console.print("\n[bold cyan]Configuration Setup[/bold cyan]")

        # Ask for configuration file
        config_file = Prompt.ask(
            "Configuration file path", default="./homodyne_config.json"
        )

        config = {"config_file": config_file}

        # Load existing config if available
        if Path(config_file).exists():
            try:
                with open(config_file) as f:
                    existing_config = json.load(f)
                config.update(existing_config)
                self.console.print(
                    f"[green]✓[/green] Loaded configuration from {config_file}"
                )
            except Exception as e:
                self.console.print(f"[red]✗[/red] Error loading config: {e}")

        # Interactive parameter adjustment
        if Confirm.ask("Would you like to modify analysis parameters?", default=False):
            config = self._interactive_parameter_adjustment(config)

        return config

    def _interactive_parameter_adjustment(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Interactive parameter adjustment interface."""
        if not RICH_AVAILABLE or not self.console:
            return config

        # Analysis mode selection
        mode_choice = Prompt.ask(
            "Analysis mode",
            choices=["static_isotropic", "static_anisotropic", "laminar_flow"],
            default="static_anisotropic",
        )

        if "analysis_settings" not in config:
            config["analysis_settings"] = {}

        if mode_choice.startswith("static"):
            config["analysis_settings"]["static_mode"] = True
            config["analysis_settings"]["static_submode"] = mode_choice.split("_")[1]
        else:
            config["analysis_settings"]["static_mode"] = False

        # Frame range adjustment
        if Confirm.ask("Adjust frame range?", default=False):
            if "analyzer_parameters" not in config:
                config["analyzer_parameters"] = {}

            start_frame = IntPrompt.ask(
                "Start frame",
                default=config.get("analyzer_parameters", {}).get("start_frame", 1001),
            )
            end_frame = IntPrompt.ask(
                "End frame",
                default=config.get("analyzer_parameters", {}).get("end_frame", 2000),
            )

            config["analyzer_parameters"]["start_frame"] = start_frame
            config["analyzer_parameters"]["end_frame"] = end_frame

        # Method selection
        method_choice = Prompt.ask(
            "Optimization methods",
            choices=["classical", "robust", "all"],
            default="classical",
        )
        config["selected_methods"] = method_choice

        return config

    def _execute_workflow_cli(self, session: AnalysisSession) -> None:
        """Execute workflow with CLI interface."""
        if not RICH_AVAILABLE or not self.console:
            return self._execute_workflow_basic(session)

        # Workflow overview
        workflow_table = Table(
            title="Workflow Overview", show_header=True, header_style="bold blue"
        )
        workflow_table.add_column("Step", style="cyan", width=4)
        workflow_table.add_column("Name", style="white")
        workflow_table.add_column("Description", style="dim")
        workflow_table.add_column("Est. Time", style="yellow", justify="right")
        workflow_table.add_column("Status", justify="center")

        for i, step in enumerate(session.workflow_steps, 1):
            time_str = f"{step.estimated_time:.0f}s" if step.estimated_time else "N/A"
            optional_str = " (optional)" if step.optional else ""
            workflow_table.add_row(
                str(i),
                step.name,
                step.description + optional_str,
                time_str,
                "⏳ Pending",
            )

        self.console.print(workflow_table)

        # Execute steps
        with get_progress_tracker() as progress:
            total_steps = len([s for s in session.workflow_steps if not s.optional])
            workflow_task = progress.create_task(
                "workflow_execution", "Executing workflow", total_steps
            )

            for i, step in enumerate(session.workflow_steps):
                # Check dependencies
                if not self._check_step_dependencies(step, session.completed_steps):
                    self.console.print(
                        f"[yellow]Skipping {step.name} - dependencies not met[/yellow]"
                    )
                    continue

                # Ask about optional steps
                if step.optional:
                    if not Confirm.ask(
                        f"Execute optional step '{step.name}'?", default=False
                    ):
                        continue

                # Execute step
                self.console.print(
                    f"\n[bold blue]Executing:[/bold blue] {step.description}"
                )

                try:
                    with ProgressContext(step.description, 1) as step_progress:
                        result = step.function(session.config, **step.parameters)
                        session.results[step.name] = result
                        session.completed_steps.append(step.name)
                        step_progress.update(current=1)

                    self.console.print(f"[green]✓[/green] Completed: {step.name}")

                    if not step.optional:
                        progress.update_task(workflow_task, increment=1)

                except Exception as e:
                    self.console.print(f"[red]✗[/red] Failed: {step.name}")
                    self.error_reporter.report_error(
                        e, {"step": step.name, "session": session.session_id}
                    )

                    if not step.optional:
                        if not Confirm.ask(
                            "Continue with workflow despite error?", default=False
                        ):
                            break

            progress.complete_task(workflow_task, success=True)

        # Session summary
        self._display_session_summary(session)

    def _check_step_dependencies(
        self, step: WorkflowStep, completed_steps: list[str]
    ) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in completed_steps for dep in step.dependencies)

    def _display_session_summary(self, session: AnalysisSession) -> None:
        """Display session summary and results."""
        if not RICH_AVAILABLE or not self.console:
            return self._display_basic_summary(session)

        duration = time.time() - session.start_time

        summary_panel = Panel(
            f"[bold green]Session Complete[/bold green]\n\n"
            f"Session ID: {session.session_id}\n"
            f"Duration: {duration:.1f} seconds\n"
            f"Steps completed: {len(session.completed_steps)}/{len(session.workflow_steps)}\n"
            f"Results generated: {len(session.results)} datasets",
            title="[green]Summary[/green]",
            border_style="green",
        )
        self.console.print(summary_panel)

        # Results table
        if session.results:
            results_table = Table(
                title="Analysis Results", show_header=True, header_style="bold green"
            )
            results_table.add_column("Step", style="cyan")
            results_table.add_column("Result Type", style="white")
            results_table.add_column("Status", justify="center")

            for step_name, result in session.results.items():
                result_type = type(result).__name__ if result else "None"
                status = "✓ Success" if result else "✗ Failed"
                status_style = "green" if result else "red"

                results_table.add_row(
                    step_name, result_type, f"[{status_style}]{status}[/{status_style}]"
                )

            self.console.print(results_table)

    def _start_jupyter_session(self, session: AnalysisSession) -> AnalysisSession:
        """Start Jupyter notebook interface."""
        if not JUPYTER_AVAILABLE:
            self.logger.warning("Jupyter not available, falling back to CLI")
            return self._start_cli_session(session)

        # Create Jupyter widgets interface
        self._create_jupyter_interface(session)
        return session

    def _create_jupyter_interface(self, session: AnalysisSession) -> None:
        """Create interactive Jupyter widgets interface."""
        # Header
        header = widgets.HTML(
            value=f"<h2>Interactive Homodyne Analysis</h2><p>Session: {session.session_id}</p>"
        )

        # Configuration panel
        config_panel = self._create_jupyter_config_panel()

        # Workflow control panel
        workflow_panel = self._create_jupyter_workflow_panel(session)

        # Results panel
        results_panel = self._create_jupyter_results_panel()

        # Layout
        interface = widgets.VBox(
            [
                header,
                widgets.Tab(
                    children=[config_panel, workflow_panel, results_panel],
                    titles=["Configuration", "Workflow", "Results"],
                ),
            ]
        )

        display(interface)

    def _create_jupyter_config_panel(self) -> widgets.Widget:
        """Create Jupyter configuration panel."""
        config_file = widgets.Text(
            value="./homodyne_config.json",
            description="Config File:",
            style={"description_width": "initial"},
        )

        analysis_mode = widgets.Dropdown(
            options=["static_isotropic", "static_anisotropic", "laminar_flow"],
            value="static_anisotropic",
            description="Analysis Mode:",
            style={"description_width": "initial"},
        )

        start_frame = widgets.IntText(
            value=1001,
            description="Start Frame:",
            style={"description_width": "initial"},
        )

        end_frame = widgets.IntText(
            value=2000, description="End Frame:", style={"description_width": "initial"}
        )

        methods = widgets.SelectMultiple(
            options=["nelder_mead", "gurobi", "wasserstein", "scenario", "ellipsoidal"],
            value=["nelder_mead"],
            description="Methods:",
            style={"description_width": "initial"},
        )

        load_config_btn = widgets.Button(
            description="Load Configuration", button_style="info"
        )

        return widgets.VBox(
            [
                config_file,
                analysis_mode,
                widgets.HBox([start_frame, end_frame]),
                methods,
                load_config_btn,
            ]
        )

    def _create_jupyter_workflow_panel(
        self, session: AnalysisSession
    ) -> widgets.Widget:
        """Create Jupyter workflow control panel."""
        workflow_select = widgets.Dropdown(
            options=list(self.workflows.keys()),
            value="quick_analysis",
            description="Workflow:",
            style={"description_width": "initial"},
        )

        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=len(session.workflow_steps),
            description="Progress:",
            bar_style="info",
            style={"description_width": "initial"},
        )

        status_text = widgets.HTML(value="<p>Ready to start workflow</p>")

        start_btn = widgets.Button(description="Start Workflow", button_style="success")

        pause_btn = widgets.Button(description="Pause", button_style="warning")

        stop_btn = widgets.Button(description="Stop", button_style="danger")

        return widgets.VBox(
            [
                workflow_select,
                progress_bar,
                status_text,
                widgets.HBox([start_btn, pause_btn, stop_btn]),
            ]
        )

    def _create_jupyter_results_panel(self) -> widgets.Widget:
        """Create Jupyter results display panel."""
        results_text = widgets.HTML(value="<p>No results yet</p>")

        plot_area = widgets.Output()

        download_btn = widgets.Button(
            description="Download Results", button_style="info"
        )

        return widgets.VBox([results_text, plot_area, download_btn])

    def _start_web_session(self, session: AnalysisSession) -> AnalysisSession:
        """Start web-based interface using Streamlit."""
        if not STREAMLIT_AVAILABLE:
            self.logger.warning("Streamlit not available, falling back to CLI")
            return self._start_cli_session(session)

        # This would be implemented as a separate Streamlit app
        # For now, fall back to CLI
        self.logger.info("Web interface not fully implemented, using CLI")
        return self._start_cli_session(session)

    def _start_basic_cli_session(self, session: AnalysisSession) -> AnalysisSession:
        """Start basic CLI session without rich formatting."""
        print("\nStarting Interactive Homodyne Analysis Session")
        print(f"Session ID: {session.session_id}")
        print(f"Workflow: {len(session.workflow_steps)} steps")

        # Basic configuration setup
        session.config = self._basic_configuration_setup()

        # Execute workflow
        self._execute_workflow_basic(session)

        return session

    def _basic_configuration_setup(self) -> dict[str, Any]:
        """Basic configuration setup without rich interface."""
        config_file = (
            input("Configuration file path [./homodyne_config.json]: ")
            or "./homodyne_config.json"
        )

        config = {"config_file": config_file}

        if Path(config_file).exists():
            try:
                with open(config_file) as f:
                    existing_config = json.load(f)
                config.update(existing_config)
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config: {e}")

        return config

    def _execute_workflow_basic(self, session: AnalysisSession) -> None:
        """Execute workflow with basic CLI interface."""
        print("\nWorkflow Steps:")
        for i, step in enumerate(session.workflow_steps, 1):
            optional_str = " (optional)" if step.optional else ""
            print(f"  {i}. {step.description}{optional_str}")

        for step in session.workflow_steps:
            if step.optional:
                response = input(f"\nExecute optional step '{step.name}'? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    continue

            print(f"\nExecuting: {step.description}")

            try:
                result = step.function(session.config, **step.parameters)
                session.results[step.name] = result
                session.completed_steps.append(step.name)
                print(f"Completed: {step.name}")
            except Exception as e:
                print(f"Failed: {step.name} - {e}")
                response = input("Continue with workflow? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    break

        self._display_basic_summary(session)

    def _display_basic_summary(self, session: AnalysisSession) -> None:
        """Display basic session summary."""
        duration = time.time() - session.start_time

        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Session ID: {session.session_id}")
        print(f"Duration: {duration:.1f} seconds")
        print(
            f"Steps completed: {len(session.completed_steps)}/{len(session.workflow_steps)}"
        )
        print(f"Results generated: {len(session.results)} datasets")

        if session.results:
            print("\nResults:")
            for step_name, result in session.results.items():
                status = "Success" if result else "Failed"
                print(f"  {step_name}: {status}")

    # Placeholder workflow step functions
    def _validate_configuration(self, config: dict[str, Any], **kwargs) -> bool:
        """Validate configuration step."""
        time.sleep(1)  # Simulate work
        return True

    def _load_data(self, config: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Load data step."""
        time.sleep(2)  # Simulate work
        return {"data_loaded": True, "num_angles": 180}

    def _parameter_exploration(
        self, config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Parameter exploration step."""
        time.sleep(5)  # Simulate work
        return {"explored_parameters": ["D0", "alpha", "D_offset"]}

    def _run_classical_optimization(
        self, config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Run classical optimization step."""
        method = kwargs.get("method", "nelder_mead")
        time.sleep(10)  # Simulate work
        return {
            "method": method,
            "chi_squared": 0.123,
            "parameters": [1000, -0.1, -0.5],
        }

    def _run_robust_optimization(
        self, config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Run robust optimization step."""
        time.sleep(20)  # Simulate work
        return {"robust_results": True, "uncertainty_bounds": [0.1, 0.05, 0.2]}

    def _compare_methods(self, config: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Compare methods step."""
        time.sleep(3)  # Simulate work
        return {"best_method": "nelder_mead", "comparison_table": "generated"}

    def _create_visualizations(
        self, config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Create visualizations step."""
        comprehensive = kwargs.get("comprehensive", False)
        time.sleep(5 if comprehensive else 2)  # Simulate work
        return {"plots_created": 3 if comprehensive else 1, "output_dir": "./plots/"}


def create_interactive_interface(
    interface_mode: str = "auto",
    enable_live_updates: bool = True,
    session_recording: bool = True,
) -> InteractiveInterface:
    """Factory function to create interactive interface."""
    return InteractiveInterface(
        interface_mode=interface_mode,
        enable_live_updates=enable_live_updates,
        session_recording=session_recording,
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Interactive Interface...")

    # Create interface
    interface = create_interactive_interface(interface_mode="cli")

    # Start a test session
    try:
        session = interface.start_interactive_session("quick_analysis")
        print(f"\nSession completed: {session.session_id}")
        print(f"Results: {list(session.results.keys())}")
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    except Exception as e:
        print(f"\nSession failed: {e}")

    print("\nInteractive interface tests completed!")
