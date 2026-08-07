"""DSPy-based agent programs for orchestration, analysis, and rendering.

This module defines the core components of the health agent:

- ReactOrchestrator: a DSPy ReAct agent that iteratively selects and executes
  domain-expert tools, adapting to what each observation returns.
- ReactAnalyzer: a DSPy ReAct agent that synthesises the accumulated tool
  outputs into the report narrative, querying artifacts through bounded
  accessors (get_metric, list_metrics, df_head, df_schema, text_read).
- Analyzer: a one-shot fallback used only if the ReAct analyzer fails.
- Reporter: a deterministic renderer (markdown + CGM figure). No LLM.
- Agent: wires the three stages together.

Heavy data objects never enter the LLM context. Tools persist DataFrames and
figures to the artifact store and return lightweight metadata plus an artifact
ID; the analyzer reads bounded views of those artifacts on demand.

Behavioural skills (markdown files) are injected at the orchestrator and
analyzer stages.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from .artifact_store import get_artifact
from .memory import build_initial_memory


# ============================================================================
# Custom Exceptions
# ============================================================================

class AnalysisError(Exception):
    """Raised when both ReactAnalyzer and the fallback Analyzer fail.

    This exception indicates a critical failure in the analysis pipeline
    where graceful degradation has been exhausted.
    """
    pass


class ReactOrchestrator(dspy.Module):
    """ReAct-based orchestrator with iterative tool selection and execution.

    Unlike the traditional Orchestrator which only plans tool execution,
    ReactOrchestrator uses DSPy's ReAct agent to both select AND execute
    tools iteratively based on observations from previous tool calls.

    Key Differences from Traditional Orchestrator:
    - Iterative: Can call tools multiple times, adjust strategy based on results
    - Self-contained: Handles both planning and execution
    - Self-correcting: Can retry or change approach on errors
    - Dynamic: Decides next tool based on actual observations, not just prompt

    Returns:
        Dictionary with 'tool_events' (already executed) instead of 'next_tools' (plan).
        The Agent class checks for 'tool_events' to skip duplicate execution.

    Attributes:
        agent: DSPy ReAct agent with converted tools
        tools: List of DSPy-compatible tool functions
    """

    def __init__(
        self,
        llm: dspy.Module,
        tools: Union[List[Callable], Dict[str, Callable]],
        max_iters: int = 10
    ):
        """Initialize ReAct orchestrator.

        Args:
            llm: Language model for ReAct agent (required)
            tools: @tool-decorated functions, or a {name: function} mapping.
            max_iters: Maximum ReAct iterations (default: 10)

        Example:
            from tools.cgm_parser.main import cgm_parser
            from tools.cgm_metrics.main import cgm_metrics
            orchestrator = ReactOrchestrator(llm, tools=[cgm_parser, cgm_metrics])
        """
        super().__init__()
        self.max_iters = max_iters

        from .tool_decorator import convert_tools_for_dspy

        # Convert tools to DSPy format
        try:
            self.tools = convert_tools_for_dspy(tools)
            print(f"✓ ReactOrchestrator loaded {len(self.tools)} tools: {[t.__name__ for t in self.tools]}")
        except Exception as e:
            print(f"⚠️  Failed to load tools for ReactOrchestrator: {e}")
            self.tools = []

        # Create ReAct agent if we have tools
        if self.tools:
            try:
                # Import ReAct (with fallback for different DSPy versions)
                try:
                    from dspy.agents import ReAct
                except ImportError:
                    from dspy.predict.react import ReAct

                # Debug: Check tool signatures
                print(f"Debug: Creating ReAct with {len(self.tools)} tools:")
                for tool in self.tools:
                    print(f"  - {tool.__name__}: {tool.__doc__[:100] if tool.__doc__ else 'No doc'}")
                    if hasattr(tool, '__signature__'):
                        print(f"    Signature: {tool.__signature__}")

                self.agent = ReAct(
                    signature="question -> answer",
                    tools=self.tools,
                    max_iters=self.max_iters,
                )
                self.agent.lm = llm
                print(f"✓ ReAct agent created successfully (max {self.max_iters} iterations)")
            except Exception as e:
                print(f"⚠️  Failed to create ReAct agent: {e}")
                import traceback
                traceback.print_exc()
                self.agent = None
        else:
            self.agent = None

    def forward(self, user_prompt: str, context_snippets: List[str], data_paths: Optional[Dict[str, str]] = None, skills_context: str = "", user_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute ReAct planning and tool execution.

        Args:
            user_prompt: User's question or request
            context_snippets: RAG-retrieved context documents
            data_paths: Available data files (e.g., {"cgm": "path/to/cgm.csv"})
            skills_context: Formatted skill instructions to prepend to context
            user_inputs: Patient metadata for tool arguments (e.g., age, sex, bmi, blood tests)

        Returns:
            Dictionary with:
            - tool_events: Already-executed tool results (skip execution in Agent)
            - report_outline: Section structure for report
            - react_used: Flag indicating ReAct was used
        """
        return self._run(user_prompt, context_snippets, data_paths, skills_context, user_inputs)

    def _run(self, user_prompt: str, context_snippets: List[str], data_paths: Optional[Dict[str, str]] = None, skills_context: str = "", user_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute ReAct workflow."""
        # Fallback if agent not initialized
        if not self.agent:
            print("⚠️  ReAct agent not available, returning empty plan")
            return {
                "next_tools": [],
                "tool_events": [],
                "report_outline": ["Overview"],
                "react_used": False,
            }

        # Enrich prompt: Skills FIRST, then RAG context
        enriched_prompt = user_prompt
        if skills_context:
            enriched_prompt += "\n\n" + skills_context
        if context_snippets:
            enriched_prompt += "\n\nRelevant Context:\n" + "\n\n".join(context_snippets[:3])

        # Add available data files information
        if data_paths:
            enriched_prompt += "\n\nAvailable Data Files:\n"
            for data_type, file_path in data_paths.items():
                enriched_prompt += f"- {data_type}: {file_path}\n"

        # Surface patient metadata so the LLM knows what data is available
        # for tool arguments (e.g., prediction tools need age, sex, bmi)
        if user_inputs:
            # Filter to non-path, non-default values that tools might need
            metadata_keys = {
                k: v for k, v in user_inputs.items()
                if not isinstance(v, str) or not v.startswith("/")
            }
            if metadata_keys:
                enriched_prompt += "\n\nAvailable Patient Metadata (use these values for tool arguments):\n"
                for key, value in metadata_keys.items():
                    enriched_prompt += f"- {key}: {value}\n"

        # Add tool usage instructions
        enriched_prompt += (
            "\n\nYou have access to tools for CGM and dietary data analysis. "
            "Use them to answer the question. "
            "When calling tools that need file paths, use the exact paths listed above. "
            "First parse the data, then compute metrics."
        )

        # Run ReAct agent with LM context
        try:
            print(f"🤖 Running ReAct agent (max {self.max_iters} iterations)...")
            print(f"   This may take 2-5 minutes depending on tool execution time...")

            import time
            start_time = time.time()

            # Use DSPy settings context to ensure LM is available to all internal modules
            import dspy

            # Enable DSPy tracing for visibility
            print(f"🔍 Enabling DSPy trace output for real-time progress...\n")

            with dspy.settings.context(lm=self.agent.lm, trace=[], show_guidelines=False):
                # Add timeout wrapper (3 minute timeout)
                import signal

                def timeout_handler(signum, frame):
                    timeout_mins = (self.max_iters * 40) // 60 + 1  # ~40s per iteration, rounded up
                    raise TimeoutError(f"ReAct execution exceeded {timeout_mins} minute timeout")

                # Set timeout only on Unix systems
                import platform
                if platform.system() != 'Windows':
                    signal.signal(signal.SIGALRM, timeout_handler)
                    timeout_secs = self.max_iters * 60  # ~60s per iteration (safe margin)
                    signal.alarm(timeout_secs)

                try:
                    result = self.agent(question=enriched_prompt)
                finally:
                    if platform.system() != 'Windows':
                        signal.alarm(0)  # Cancel timeout

            elapsed = time.time() - start_time
            print(f"✓ ReAct agent completed in {elapsed:.1f} seconds\n")

            # Print DSPy trace if available
            if dspy.settings.trace:
                print(f"📊 DSPy captured {len(dspy.settings.trace)} trace events")

            # Show ReAct trajectory (what it thought and did)
            self._print_trajectory(result)

            # Parse ReAct trajectory into tool_events format
            tool_events = self._parse_react_trajectory(result)

            # Count iterations from trajectory
            iterations_used = 0
            if hasattr(result, 'trajectory') and isinstance(result.trajectory, dict):
                for key in result.trajectory.keys():
                    if key.startswith('thought_'):
                        try:
                            iter_num = int(key.split('_')[-1]) + 1
                            iterations_used = max(iterations_used, iter_num)
                        except (ValueError, IndexError):
                            pass

            print(f"✓ ReAct completed with {len(tool_events)} tool calls in {iterations_used}/{self.max_iters} iterations")

            return {
                "next_tools": [],  # Empty - tools already executed by ReAct
                "tool_events": tool_events,  # Already-executed results
                "report_outline": ["Overview", "Key Findings", "Analysis"],
                "react_used": True,
                "iterations_used": iterations_used,
                "max_iters": self.max_iters,
            }

        except TimeoutError as e:
            timeout_mins = (self.max_iters * 60) // 60  # Convert seconds to minutes
            print(f"⏱️  ReAct execution timed out after {timeout_mins} minutes")
            print(f"   This suggests a tool is hanging or the LLM is not responding")
            raise

        except Exception as e:
            print(f"⚠️  ReAct execution failed: {type(e).__name__}: {e}")
            raise

    def _print_trajectory(self, result: Any) -> None:
        """Print ReAct trajectory in a readable format showing thoughts and actions.

        Args:
            result: ReAct result object with trajectory attribute
        """
        if not hasattr(result, 'trajectory'):
            print("⚠️  No trajectory found in result")
            return

        trajectory = result.trajectory
        if not trajectory:
            print("⚠️  Empty trajectory")
            return

        print("=" * 80)
        print("📝 ReAct Trajectory (Reasoning + Actions)")
        print("=" * 80)

        # Handle dict-based trajectory
        if isinstance(trajectory, dict):
            # Find max iteration number
            max_iter = 0
            for key in trajectory.keys():
                if '_' in key:
                    try:
                        iter_num = int(key.split('_')[-1])
                        max_iter = max(max_iter, iter_num)
                    except (ValueError, IndexError):
                        pass

            # Print each iteration
            for i in range(max_iter + 1):
                thought = trajectory.get(f'thought_{i}')
                tool_name = trajectory.get(f'tool_name_{i}')
                tool_args = trajectory.get(f'tool_args_{i}')
                observation = trajectory.get(f'observation_{i}')

                if thought or tool_name:
                    print(f"\n🔄 Iteration {i + 1}:")

                if thought:
                    # Truncate long thoughts
                    thought_str = str(thought)
                    if len(thought_str) > 200:
                        thought_str = thought_str[:200] + "..."
                    print(f"  💭 Thought: {thought_str}")

                if tool_name:
                    print(f"  🔧 Tool: {tool_name}")
                    if tool_args and isinstance(tool_args, dict):
                        print(f"  📥 Args:")
                        for key, val in tool_args.items():
                            # Truncate long values
                            val_str = str(val)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            print(f"      {key}: {val_str}")

                if observation:
                    # Truncate long observations
                    obs_str = str(observation)
                    if len(obs_str) > 300:
                        obs_str = obs_str[:300] + "..."
                    print(f"  📤 Observation: {obs_str}")

        else:
            # Handle list-based trajectory (other DSPy versions)
            for i, step in enumerate(trajectory):
                print(f"\n🔄 Iteration {i + 1}:")
                print(f"  {step}")

        print("\n" + "=" * 80 + "\n")

    def _parse_observation_to_output(self, observation: Any) -> Dict[str, Any]:
        """Parse observation (from DSPy tool) into structured output format.

        DSPy tools return observation strings formatted by to_dspy_tool().
        These strings contain JSON metadata that we need to extract.

        Expected format (from to_dspy_tool):
            ✓ Executed tool_name
            Metadata: {...}
            Artifacts: {...}
            Warnings: ...

        Args:
            observation: Tool observation (string or dict)

        Returns:
            Structured output dict with status, summary, metadata, artifacts
        """
        # If already a dict, return as-is
        if isinstance(observation, dict):
            return observation

        # Parse observation string
        if observation and isinstance(observation, str):
            try:
                # Try to extract JSON metadata from observation
                if '{' in observation and '}' in observation:
                    lines = observation.split('\n')
                    metadata = None
                    artifacts = {}

                    # Look for Metadata: and Artifacts: sections
                    for i, line in enumerate(lines):
                        if line.startswith('Metadata:'):
                            # Extract JSON starting from the opening brace
                            json_start_idx = line.find('{')
                            if json_start_idx != -1:
                                # Collect JSON lines including the opening brace
                                json_lines = [line[json_start_idx:]]
                                for j in range(i + 1, len(lines)):
                                    if lines[j].startswith('Artifacts:') or lines[j].startswith('Warnings:'):
                                        break
                                    json_lines.append(lines[j])

                                # Parse metadata JSON
                                try:
                                    metadata_json = '\n'.join(json_lines)
                                    metadata = json.loads(metadata_json)
                                except json.JSONDecodeError as e:
                                    print(f"Debug: Failed to parse metadata JSON: {e}")

                        elif line.startswith('Artifacts:'):
                            # Extract JSON starting from the opening brace
                            json_start_idx = line.find('{')
                            if json_start_idx != -1:
                                # Collect JSON lines including the opening brace
                                json_lines = [line[json_start_idx:]]
                                for j in range(i + 1, len(lines)):
                                    if lines[j].startswith('Warnings:'):
                                        break
                                    json_lines.append(lines[j])

                                # Parse artifacts JSON
                                try:
                                    artifacts_json = '\n'.join(json_lines)
                                    artifacts = json.loads(artifacts_json)
                                except json.JSONDecodeError as e:
                                    print(f"Debug: Failed to parse artifacts JSON: {e}")

                    # Build output structure matching tool decorator format
                    if metadata is not None or artifacts:
                        return {
                            "status": "error" if "✗" in observation else "ok",
                            "summary": lines[0] if lines else "",
                            "metadata": metadata,
                            "artifacts": artifacts,
                        }

            except Exception as e:
                # Parsing failed, return observation as summary
                print(f"Debug: Failed to parse observation JSON: {e}")

        # Fallback: return as simple summary
        return {
            "status": "error" if observation and "✗" in str(observation) else "ok",
            "summary": str(observation)[:200] if observation else "",
        }

    def _parse_react_trajectory(self, result: Any) -> List[Dict[str, Any]]:
        """Parse ReAct trajectory into tool_events format.

        ReAct stores its execution history in a trajectory. We need to extract:
        - Tool name
        - Tool arguments
        - Tool output (observation)

        And convert to the format expected by Analyzer/Reporter:
        [{"tool_name": str, "args": dict, "output": dict}, ...]

        Args:
            result: ReAct result object with trajectory attribute

        Returns:
            List of tool events in standard format
        """
        tool_events = []

        # ReAct stores trajectory in different formats depending on version
        # Try multiple access patterns
        trajectory = None
        if hasattr(result, 'trajectory'):
            trajectory = result.trajectory
            print(f"Debug: Found trajectory (length: {len(trajectory) if trajectory else 0})")
        elif hasattr(result, 'history'):
            trajectory = result.history
            print(f"Debug: Found history (length: {len(trajectory) if trajectory else 0})")
        elif hasattr(result, 'trace'):
            trajectory = result.trace
            print(f"Debug: Found trace (length: {len(trajectory) if trajectory else 0})")

        if not trajectory:
            print("⚠️  No trajectory found in ReAct result")
            print(f"Debug: Result attributes: {list(result.__dict__.keys()) if hasattr(result, '__dict__') else 'No __dict__'}")
            return tool_events

        # Handle trajectory as dict (DSPy format: {'thought_0': ..., 'tool_name_0': ..., 'tool_args_0': ..., 'observation_0': ...})
        if isinstance(trajectory, dict):
            print(f"Debug: Trajectory is dict with {len(trajectory)} keys")
            print(f"Debug: Trajectory keys (first 10): {list(trajectory.keys())[:10]}")

            # Parse dict-based trajectory
            # Find max iteration number
            max_iter = 0
            for key in trajectory.keys():
                if '_' in key:
                    try:
                        iter_num = int(key.split('_')[-1])
                        max_iter = max(max_iter, iter_num)
                    except (ValueError, IndexError):
                        pass

            print(f"Debug: Found {max_iter + 1} iterations in trajectory")

            # Extract tool calls from each iteration
            for i in range(max_iter + 1):
                tool_name = trajectory.get(f'tool_name_{i}')
                tool_args = trajectory.get(f'tool_args_{i}')
                observation = trajectory.get(f'observation_{i}')

                print(f"Debug: Iteration {i}: tool_name={tool_name}, has_args={tool_args is not None}, has_obs={observation is not None}")

                if tool_name and tool_name != 'finish':
                    # Parse observation string to extract structured data
                    tool_output = self._parse_observation_to_output(observation)

                    # Valid tool call
                    tool_events.append({
                        "tool_name": tool_name,
                        "args": tool_args if isinstance(tool_args, dict) else {},
                        "output": tool_output,
                    })

            return tool_events

        # Handle trajectory as list (fallback for other DSPy versions)
        print(f"Debug: Trajectory is {type(trajectory)} with {len(trajectory)} items")
        for i, step in enumerate(list(trajectory)[:3]):  # Show first 3 steps, convert to list if needed
            print(f"Debug: Step {i}: {type(step)} - {step if len(str(step)) < 200 else str(step)[:200]+'...'}")

        # Parse each step in trajectory
        for i, step in enumerate(trajectory):
            try:
                # Extract tool information from step
                # Format varies by DSPy version, try multiple patterns
                tool_name = None
                tool_args = {}
                tool_output = None

                # Pattern 1: step is dict with 'action' and 'observation'
                if isinstance(step, dict):
                    tool_name = step.get('action') or step.get('tool')
                    tool_args = step.get('action_input', {}) or step.get('args', {})
                    observation = step.get('observation') or step.get('result')

                    # Use common parsing logic
                    tool_output = self._parse_observation_to_output(observation)

                # Pattern 2: step has attributes
                elif hasattr(step, 'action'):
                    tool_name = getattr(step, 'action', None)
                    tool_args = getattr(step, 'action_input', {})
                    observation = getattr(step, 'observation', None)

                    # Use common parsing logic
                    tool_output = self._parse_observation_to_output(observation)

                # Only add if we found a tool call
                if tool_name:
                    tool_events.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "output": tool_output or {"status": "ok", "summary": ""},
                    })

            except Exception as e:
                print(f"⚠️  Failed to parse trajectory step {i}: {e}")
                continue

        return tool_events


class Reporter(dspy.Module):
    """Deterministic renderer for the analyzer's report. No LLM involvement.

    The ReactAnalyzer produces the complete report narrative; the Reporter
    writes it to disk together with a daily glucose-profile figure rendered
    from the parsed CGM artifact. Keeping this stage deterministic means the
    figure can never disagree with the data it was drawn from.

    Attributes:
        output_dir: Directory where the report and figure are written.
    """

    # Consensus CGM target range in mg/dL (Battelino et al., Diabetes Care 2019)
    TARGET_RANGE_MGDL = (70.0, 180.0)

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the Reporter.

        Args:
            output_dir: Directory for the report and figure. Defaults to a
                temporary directory.
        """
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def forward(
        self,
        analysis_text: str,
        tool_events: List[Dict[str, Any]],
        filename: str = "report.md",
    ) -> str:
        """Render the report.

        Args:
            analysis_text: Complete report narrative from the analyzer.
            tool_events: Tool outputs, used to locate the parsed CGM artifact.
            filename: Name of the markdown file to write.

        Returns:
            Path to the written markdown report.
        """
        return self._run(analysis_text, tool_events, filename)

    def _find_cgm_frame(self, tool_events: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Locate the parsed CGM DataFrame from the cgm_parser tool event.

        Args:
            tool_events: Tool outputs from the orchestrator.

        Returns:
            The parsed CGM DataFrame, or None if it is unavailable.
        """
        parser_event = next(
            (e for e in tool_events if e.get("tool_name") == "cgm_parser"), None
        )
        if not parser_event:
            print("Note: no cgm_parser event found, skipping figure")
            return None

        output = parser_event.get("output", {}) or {}
        metadata = output.get("metadata") or {}
        artifacts = output.get("artifacts") or {}
        aid = metadata.get("cgm_aid") or artifacts.get("cgm_aid")
        if not aid:
            print("Note: no cgm_aid in parser output, skipping figure")
            return None

        try:
            df = get_artifact(aid, expected_prefix="cgm")
        except Exception as e:
            print(f"Warning: could not load CGM artifact for plotting: {e}")
            return None

        if df is None or df.empty:
            print("Note: CGM DataFrame is empty, skipping figure")
            return None
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Find the timestamp and glucose columns in a parsed CGM frame."""
        time_col = next((c for c in df.columns if "time" in c.lower()), None)
        glucose_col = next(
            (c for c in df.columns if c.lower() in {"gl", "glucose"}), None
        ) or next((c for c in df.columns if "glucose" in c.lower()), None)
        return time_col, glucose_col

    def _plot_daily_profile(
        self, df: pd.DataFrame, filename: str = "cgm_daily.png"
    ) -> Optional[Path]:
        """Overlay every recorded day's glucose trace on a 24-hour axis.

        Args:
            df: Parsed CGM DataFrame.
            filename: Name of the PNG file to write.

        Returns:
            Path to the written figure, or None if it could not be produced.
        """
        time_col, glucose_col = self._resolve_columns(df)
        if time_col is None or glucose_col is None:
            print(
                f"Warning: could not identify time/glucose columns in {list(df.columns)}, "
                "skipping figure"
            )
            return None

        try:
            timestamps = pd.to_datetime(df[time_col], errors="coerce")
            frame = pd.DataFrame({
                "hour": timestamps.dt.hour + timestamps.dt.minute / 60.0,
                "date": timestamps.dt.date,
                "glucose": pd.to_numeric(df[glucose_col], errors="coerce"),
            }).dropna()
            if frame.empty:
                print("Note: no usable CGM rows for plotting, skipping figure")
                return None

            fig, ax = plt.subplots(figsize=(10, 4.5))

            # One faint trace per day, then the pointwise median across days.
            for _, day_rows in frame.groupby("date"):
                day_rows = day_rows.sort_values("hour")
                ax.plot(day_rows["hour"], day_rows["glucose"],
                        color="steelblue", alpha=0.25, linewidth=1)

            median = frame.groupby(frame["hour"].round())["glucose"].median()
            ax.plot(median.index, median.values, color="navy", linewidth=2.5,
                    label="Median across days")

            low, high = self.TARGET_RANGE_MGDL
            ax.axhspan(low, high, color="seagreen", alpha=0.10,
                       label=f"Target range {low:.0f}-{high:.0f} mg/dL")

            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 3))
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Glucose (mg/dL)")
            ax.set_title(f"Daily glucose profile ({frame['date'].nunique()} days overlaid)")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()

            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return output_path

        except Exception as e:
            print(f"Warning: figure generation failed: {e}")
            return None

    def _run(
        self,
        analysis_text: str,
        tool_events: List[Dict[str, Any]],
        filename: str = "report.md",
    ) -> str:
        """Write the markdown report and, where possible, the CGM figure."""
        body = analysis_text or ""

        df = self._find_cgm_frame(tool_events)
        figure_path = self._plot_daily_profile(df) if df is not None else None
        if figure_path is not None:
            body += (
                "\n\n## Daily Glucose Profile\n\n"
                f"![Daily glucose profile]({figure_path.name})\n"
            )

        report_path = self.output_dir / filename
        report_path.write_text(body, encoding="utf-8")
        return str(report_path)


class AnalyzerSig(dspy.Signature):
    """Analyze CGM results and provide interpretation and recommendations.

    The Reporter already presented the raw data and metrics.
    Your job is to:
    - Interpret what the metrics mean in context of the user's question
    - Identify key patterns, trends, or concerns
    - Provide actionable recommendations
    - Use evidence from context documents when relevant

    CRITICAL: Check the metric_units field in tool_events metadata to correctly interpret
    metric values. Refer to metric_units[metric_name] for the units/format specification.
    Always use values according to their documented units without arbitrary conversion.

    Format requirements:
    - Start with ## Interpretation heading (continue the report, don't add # heading)
    - Add ## Recommendations section
    - Be specific to the user's question
    - Avoid repeating raw numbers already shown in Key Metrics section
    """

    user_prompt = dspy.InputField(desc="Original user question or goal")
    tool_events = dspy.InputField(desc="Full tool outputs including daily data and metadata")
    context_docs = dspy.InputField(desc="Documents retrieved as supporting context")
    skills_context = dspy.InputField(desc="Behavioral guidelines and constraints to follow", default="")
    analysis_markdown = dspy.OutputField(desc="Interpretation and recommendations (## headings)")


class Analyzer(dspy.Module):
    """Provides interpretation and recommendations for CGM results.

    This module analyzes CGM metrics and tool outputs to provide meaningful
    interpretation, identify patterns, and offer actionable recommendations.
    It uses scientific context from RAG when available.

    Attributes:
        predict: Optional DSPy prediction module for LLM-based analysis.
    """

    def __init__(self, llm: Optional[dspy.Module] = None):
        """Initialize the Analyzer.

        Args:
            llm: Optional language model for analysis. If None, returns empty string.
        """
        super().__init__()
        # Note: Don't pass llm= to Predict() constructor in DSPy 3.x - causes JSON serialization errors
        # Instead, create predictor and set .lm attribute directly for per-module LLM support
        if llm is not None:
            self.predict = dspy.Predict(AnalyzerSig)
            self.predict.lm = llm
        else:
            self.predict = None

    def forward(
        self,
        user_prompt: str,
        tool_events: List[Dict[str, Any]],
        context_docs: List[Dict[str, Any]],
        skills_context: str = "",
    ) -> str:
        return self._run(user_prompt, tool_events, context_docs, skills_context)

    def _run(
        self,
        user_prompt: str,
        tool_events: List[Dict[str, Any]],
        context_docs: List[Dict[str, Any]],
        skills_context: str = "",
    ) -> str:
        if self.predict is not None:
            try:
                raw = self.predict(
                    user_prompt=user_prompt,
                    tool_events=tool_events,
                    context_docs=context_docs,
                    skills_context=skills_context,
                )
                if isinstance(raw.analysis_markdown, str) and raw.analysis_markdown.strip():
                    return raw.analysis_markdown
            except Exception as e:
                print(f"⚠️  Analyzer LLM call failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # Simple fallback when LLM is unavailable
        return ""


class ReactAnalyzer(dspy.Module):
    """ReAct-based analyzer with iterative artifact exploration.

    Uses bounded accessor tools to explore artifacts when facts alone
    are insufficient for deep analysis.

    Key Features:
    - Starts with facts (scalars from tool metadata)
    - Explores artifacts iteratively when needed
    - Uses bounded accessors (get_metric, list_metrics, df_head, df_schema)
    - Context includes artifact preview map

    Design Philosophy:
    - Facts first: Start with scalars, explore only when needed
    - Bounded access: All accessors enforce size limits
    - Artifact map: Agent knows what's available before exploring

    Attributes:
        agent: DSPy ReAct agent with accessor tools
        tools: List of accessor tool functions
        max_iters: Maximum ReAct iterations
    """

    def __init__(self, llm: dspy.Module, max_iters: int = 5, fallback_analyzer: Optional['Analyzer'] = None):
        """Initialize ReactAnalyzer with fallback support.

        Args:
            llm: Language model for ReAct agent (required)
            max_iters: Maximum ReAct iterations (default: 5)
            fallback_analyzer: Optional traditional Analyzer to fall back to on errors.
                             If None, creates a new Analyzer instance.
        """
        super().__init__()
        self.max_iters = max_iters

        # Create or use provided fallback analyzer (Python Zen: errors shouldn't pass silently)
        self.fallback = fallback_analyzer or Analyzer(llm)

        # Import accessor tools
        from .artifact_accessors import get_metric, list_metrics, df_head, df_schema, text_read

        # Convert accessors to DSPy tool format
        self.tools = [get_metric, list_metrics, df_head, df_schema, text_read]

        # Create ReAct agent
        try:
            try:
                from dspy.agents import ReAct
            except ImportError:
                from dspy.predict.react import ReAct

            self.agent = ReAct(
                signature="analysis_context -> analysis_markdown",
                tools=self.tools,
                max_iters=self.max_iters,
            )
            self.agent.lm = llm
            print(f"✓ ReactAnalyzer initialized with {len(self.tools)} accessor tools (max {self.max_iters} iterations)")

        except Exception as e:
            print(f"⚠️  Failed to create ReactAnalyzer: {e}")
            import traceback
            traceback.print_exc()
            self.agent = None

    def forward(
        self,
        user_prompt: str,
        tool_events: List[Dict[str, Any]],
        context_docs: List[Dict[str, Any]],
        artifact_map: Optional[Dict[str, Dict[str, Any]]] = None,
        skills_context: str = "",
    ) -> str:
        """Run ReAct-based analysis with artifact exploration.

        Args:
            user_prompt: User's question or request
            tool_events: Tool execution results
            context_docs: RAG-retrieved context documents
            artifact_map: Map of available artifacts with previews
            skills_context: Formatted skill instructions to include in context

        Returns:
            Analysis markdown with interpretation and recommendations
        """
        return self._run(user_prompt, tool_events, context_docs, artifact_map or {}, skills_context)

    def _run(
        self,
        user_prompt: str,
        tool_events: List[Dict[str, Any]],
        context_docs: List[Dict[str, Any]],
        artifact_map: Dict[str, Dict[str, Any]],
        skills_context: str = "",
    ) -> str:
        """Execute ReAct-based analysis."""
        if not self.agent:
            print("⚠️  ReactAnalyzer agent not available, returning empty analysis")
            return ""

        # Build rich context with facts + artifact map + skills
        context = self._build_context(user_prompt, tool_events, context_docs, artifact_map, skills_context)

        # Run ReAct agent with proper DSPy context
        try:
            print(f"🤖 Running ReactAnalyzer (max {self.max_iters} iterations)...")
            import time
            start_time = time.time()

            # Capture trace for observability
            trace = []

            # Ensure LM is available in DSPy settings context
            import dspy
            with dspy.settings.context(lm=self.agent.lm, trace=trace, show_guidelines=False):
                result = self.agent(analysis_context=context)

            elapsed = time.time() - start_time

            # Extract observability info from result
            tool_calls, iterations_used = self._extract_tool_calls(result, trace)

            print(f"✓ ReactAnalyzer completed in {elapsed:.2f}s")
            print(f"  Iterations: {iterations_used}/{self.max_iters}")
            if tool_calls:
                print(f"  Tool calls: {len(tool_calls)}")
                for i, call in enumerate(tool_calls[:5], 1):  # Show first 5
                    print(f"    {i}. {call['tool']}({call['args']}) → {call['result_preview']}")
                if len(tool_calls) > 5:
                    print(f"    ... and {len(tool_calls) - 5} more")
            else:
                print(f"  ⚠️  No tool calls extracted (result structure may not match expectations)")

            # Store last execution info for provenance tracking
            self.last_iterations_used = iterations_used
            self.last_tool_calls = len(tool_calls)
            self.last_tool_calls_detail = tool_calls  # Full details for trace

            return result.analysis_markdown

        except Exception as e:
            print(f"⚠️  ReactAnalyzer execution failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

            # Graceful degradation: Fall back to traditional analyzer
            print(f"⚠️  Falling back to traditional Analyzer...")
            try:
                return self.fallback(
                    user_prompt=user_prompt,
                    tool_events=tool_events,
                    context_docs=context_docs,
                    skills_context=skills_context,
                )
            except Exception as fallback_error:
                # Both analyzers failed - this is unexpected, raise
                print(f"❌ Fallback analyzer also failed: {fallback_error}")
                raise AnalysisError(
                    f"Both ReactAnalyzer and fallback Analyzer failed. "
                    f"ReactAnalyzer error: {e}. Fallback error: {fallback_error}"
                ) from e

    def _build_context(
        self,
        user_prompt: str,
        tool_events: List[Dict[str, Any]],
        context_docs: List[Dict[str, Any]],
        artifact_map: Dict[str, Dict[str, Any]],
        skills_context: str = "",
    ) -> str:
        """Build rich context including artifact map and skills."""
        lines = []

        # 0. Skills (FIRST - before everything else)
        if skills_context:
            lines.append(skills_context)
            lines.append("")

        # 1. User question
        lines.append("# USER QUESTION")
        lines.append(user_prompt)
        lines.append("")

        # 2. Facts from metadata (extracted from tool outputs)
        lines.append("# KEY FACTS")
        lines.append("Facts extracted from tool outputs:")
        lines.append("")

        for event in tool_events:
            tool_name = event.get("tool_name")
            metadata = event.get("output", {}).get("metadata", {})

            # Extract scalars from metadata
            facts = self._extract_scalars(metadata)

            if facts:
                lines.append(f"## {tool_name}:")
                for key, value in facts.items():
                    if isinstance(value, float):
                        lines.append(f"  - {key}: {value:.2f}")
                    else:
                        lines.append(f"  - {key}: {value}")
                lines.append("")

        # 3. Artifact map (THE KEY PART)
        if artifact_map:
            lines.append("# ARTIFACTS AVAILABLE FOR EXPLORATION")
            lines.append("You can use accessor tools to explore these artifacts:")
            lines.append("")

            for artifact_id, info in artifact_map.items():
                lines.append(f"## Artifact: {artifact_id}")
                lines.append(f"   Source: {info['source_tool']} ({info['artifact_name']})")
                lines.append(f"   Type: {info['type']}")

                preview = info.get("preview", {})

                if info["type"] == "json":
                    tools = info.get("tools", [])
                    if "get_metric" in tools:  # dict artifact
                        lines.append(f"   Available keys ({preview.get('total_count', '?')} total):")
                        for key in preview.get("sample_keys", [])[:5]:
                            lines.append(f"     - {key}")
                        if preview.get("total_count", 0) > 5:
                            lines.append(f"     ... and {preview['total_count'] - 5} more")
                        lines.append(f"   → Use: text_read('{artifact_id}') to see full content")
                        lines.append(f"   → Use: get_metric('{artifact_id}', 'key') for scalar values")
                    else:  # list artifact
                        lines.append(f"   JSON list artifact")
                        lines.append(f"   → Use: text_read('{artifact_id}') to see full content")

                elif info["type"] == "dataframe":
                    lines.append(f"   Shape: {preview.get('shape', '?')}")
                    lines.append(f"   Columns: {', '.join(preview.get('columns', []))}")
                    lines.append(f"   Size: {preview.get('memory_mb', '?')} MB")
                    lines.append(f"   → Use: df_head('{artifact_id}', n=10) to preview")
                    lines.append(f"   → Use: df_schema('{artifact_id}') for details")

                elif info["type"] == "plot":
                    lines.append(f"   Image/visualization artifact")

                lines.append("")

        # 4. RAG context
        if context_docs:
            lines.append("# SCIENTIFIC CONTEXT")
            lines.append("Relevant context from literature:")
            lines.append("")
            for i, doc in enumerate(context_docs[:3], 1):
                snippet = doc["text"][:200] + "..."
                lines.append(f"{i}. {snippet}")
            lines.append("")

        # 5. Task instructions
        lines.append("# YOUR TASK")
        lines.append("Analyze the user's question using facts and artifacts.")
        lines.append("")
        lines.append("APPROACH:")
        lines.append("1. Start with the KEY FACTS (already extracted)")
        lines.append("2. Use accessor tools ONLY if you need more detail from artifacts")
        lines.append("3. Generate ## Interpretation and ## Recommendations sections")
        lines.append("")
        if skills_context:
            # A report-structure skill is active and owns the output format.
            # Imposing a second format here produces both, and the report ends
            # up with stray headings that belong to neither.
            lines.append("FORMAT:")
            lines.append("- Follow the report structure defined in the skills above, exactly")
            lines.append("- Be specific and actionable")
        else:
            lines.append("FORMAT:")
            lines.append("- Start with ## Interpretation heading")
            lines.append("- Add ## Recommendations section")
            lines.append("- Be specific and actionable")

        return "\n".join(lines)

    def _extract_scalars(self, metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively extract scalar values from nested metadata dicts."""
        scalars = {}
        for key, value in metadata.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if key.endswith("_aid"):
                continue
            if isinstance(value, dict):
                nested_scalars = self._extract_scalars(value, prefix=full_key)
                scalars.update(nested_scalars)
            elif isinstance(value, (int, float, str, bool)):
                scalars[full_key] = value
        return scalars

    def _extract_tool_calls(self, result, trace: List = None) -> tuple[List[Dict[str, Any]], int]:
        """Extract tool calls and iteration count from ReAct result.

        Parses the DSPy ReAct result object to extract:
        - Tool name
        - Arguments passed
        - Result preview (truncated)
        - Iteration count

        Args:
            result: DSPy ReAct result object
            trace: Optional DSPy trace list (unused in current implementation)

        Returns:
            Tuple of (tool_calls list, iterations_used int)
        """
        tool_calls = []
        iterations_used = 0

        try:
            # Approach 1: Check for trajectory attribute (current DSPy ReAct format)
            if hasattr(result, 'trajectory'):
                trajectory = result.trajectory

                if isinstance(trajectory, dict):
                    # DSPy ReAct stores trajectory as flat dict with keys like:
                    # thought_0, tool_name_0, tool_args_0, observation_0, thought_1, ...
                    # Group by step number
                    steps = {}
                    for key, value in trajectory.items():
                        # Extract step number from key (e.g., "tool_name_0" -> 0)
                        parts = key.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            step_num = int(parts[1])
                            field_name = parts[0]

                            if step_num not in steps:
                                steps[step_num] = {}
                            steps[step_num][field_name] = value

                    iterations_used = len(steps)

                    # Extract tool calls from steps (excluding 'finish')
                    for step_num in sorted(steps.keys()):
                        step = steps[step_num]
                        tool_name = step.get('tool_name', '')
                        tool_args = step.get('tool_args', '')
                        observation = step.get('observation', '')

                        if tool_name and tool_name.lower() != 'finish':
                            # Safely convert observation to string and truncate
                            obs_str = str(observation) if observation else ''
                            result_preview = obs_str[:100] + ('...' if len(obs_str) > 100 else '')

                            tool_calls.append({
                                'tool': tool_name,
                                'args': str(tool_args) if tool_args else '',
                                'result_preview': result_preview
                            })
                elif isinstance(trajectory, list):
                    print(f"  DEBUG: Trajectory has {len(trajectory)} items")
                    for item in trajectory:
                        # Each trajectory item might be a dict with 'action' and 'observation'
                        if isinstance(item, dict):
                            action = item.get('action', '')
                            observation = item.get('observation', '')

                            # Parse action string (usually "tool_name(args)")
                            tool_call = self._parse_action_string(action, observation)
                            if tool_call:
                                tool_calls.append(tool_call)
                else:
                    print(f"  DEBUG: Trajectory is unexpected type: {type(trajectory)}")

            # Approach 2: Check for individual action/observation attributes
            elif hasattr(result, 'actions') and hasattr(result, 'observations'):
                actions = result.actions if isinstance(result.actions, list) else [result.actions]
                observations = result.observations if isinstance(result.observations, list) else [result.observations]

                for action, observation in zip(actions, observations):
                    tool_call = self._parse_action_string(str(action), str(observation))
                    if tool_call:
                        tool_calls.append(tool_call)

            # Approach 3: Inspect result dict if it's dictionary-like
            elif hasattr(result, '__dict__'):
                result_dict = result.__dict__

                # Look for tool-related keys
                for key, value in result_dict.items():
                    if 'action' in key.lower() or 'tool' in key.lower():
                        if isinstance(value, (list, tuple)):
                            for item in value:
                                tool_call = self._parse_action_string(str(item), "")
                                if tool_call:
                                    tool_calls.append(tool_call)

        except Exception as e:
            # Silently fail - observability is nice-to-have, not critical
            print(f"  ⚠️  Could not extract tool calls: {e}")

        return tool_calls, iterations_used

    def _parse_action_string(self, action: str, observation: str) -> Optional[Dict[str, Any]]:
        """Parse action string like 'get_metric(artifact_id, metric_name)' into structured data.

        Args:
            action: Action string (e.g., "get_metric('metrics_abc', 'cgm_mean')")
            observation: Observation/result string

        Returns:
            Dict with 'tool', 'args', 'result_preview' or None if unparseable
        """
        import re

        # Skip non-tool actions (like "Finish" or plain text)
        if not action or 'Finish' in action or '(' not in action:
            return None

        try:
            # Extract tool name and arguments using regex
            # Pattern: tool_name(arg1, arg2, ...)
            match = re.match(r'(\w+)\((.*)\)', action.strip())
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)

                # Truncate observation for preview
                result_preview = observation[:100] if observation else "N/A"
                if len(observation) > 100:
                    result_preview += "..."

                return {
                    'tool': tool_name,
                    'args': args_str,
                    'result_preview': result_preview
                }

        except Exception:
            pass

        return None


class Agent(dspy.Module):
    """Complete health agent workflow: orchestrate, analyse, render.

    Three stages:

    1. ReactOrchestrator iteratively selects and executes domain-expert tools.
       Each tool returns lightweight metadata to the orchestrator and persists
       heavy objects (DataFrames, figures) to the artifact store.
    2. ReactAnalyzer synthesises those outputs into the report narrative,
       querying artifacts through bounded accessors. It never receives a full
       dataset, only bounded views.
    3. Reporter deterministically renders the narrative, without an LLM.

    Behavioural skills are injected at stages 1 and 2.

    Usage:
        from tools.cgm_parser.main import cgm_parser
        from tools.cgm_metrics.main import cgm_metrics

        agent = Agent(
            llm=lm,
            tools=[cgm_parser, cgm_metrics],
            output_dir=Path("out"),
            skills=["clinical-language-guard"],
        )
        memory = agent(
            user_prompt="Assess my glucose control",
            data_paths={"cgm": "data/synthetic_001/cgm.csv"},
        )

    Attributes:
        orchestrator: ReactOrchestrator over the supplied tools.
        analyzer: ReactAnalyzer over the bounded artifact accessors.
        reporter: Deterministic markdown + figure renderer.
        skill_loader: Loads and formats skills for prompt injection.
    """

    def __init__(
        self,
        llm: dspy.Module,
        tools: Union[List[Callable], Dict[str, Callable]],
        output_dir: Optional[Path] = None,
        max_iters: int = 10,
        analyzer_max_iters: int = 8,
        skills: Optional[List[str]] = None,
    ):
        """Initialize the Agent with all components.

        Args:
            llm: Language model for the orchestrator and analyzer (required).
            tools: @tool-decorated functions the orchestrator may call, or a
                {name: function} mapping. There is no implicit tool registry:
                whatever is passed here is exactly what the agent can use.
            output_dir: Directory for the rendered report and figure.
            max_iters: Maximum ReactOrchestrator iterations.
            analyzer_max_iters: Maximum ReactAnalyzer iterations.
            skills: Skill names to activate, e.g. ["clinical-language-guard"].
        """
        super().__init__()
        self.llm = llm
        self.tools = tools
        self.max_iters = max_iters

        self.orchestrator = ReactOrchestrator(llm, tools=tools, max_iters=max_iters)
        self.analyzer = ReactAnalyzer(llm, max_iters=analyzer_max_iters)
        self.reporter = Reporter(output_dir)

        from .skill_loader import SkillLoader
        self.skill_loader = SkillLoader()
        self.skills = skills or []
        if self.skills:
            loaded = self.skill_loader.load_skills(self.skills)
            print(f"✓ Loaded {len(loaded)} skills: {[s.name for s in loaded]}")

    def _build_artifact_preview_map(
        self,
        tool_events: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build map of all artifacts with previews using file-based type inference.

        For each artifact referenced in tool_events:
        1. Extract artifact ID (file path) from metadata (_aid fields)
        2. Infer type from file extension via artifact_info()
        3. Use appropriate accessors for preview
        4. Mark explorable status and list available tools

        Returns:
            Dict mapping artifact_id -> preview info with type, explorable, tools

        Example:
            {
                "metrics_xyz456": {
                    "type": "json",
                    "explorable": True,
                    "tools": ["text_read", "get_metric", "list_metrics"],
                    "source_tool": "cgm_metrics",
                    "artifact_name": "global_metrics_aid",
                    "preview": {"sample_keys": [...], "total_count": 42}
                },
                "df_meals_abc123": {
                    "type": "dataframe",
                    "explorable": True,
                    "tools": ["df_head", "df_schema"],
                    "source_tool": "meal_parser",
                    "artifact_name": "meal_aid",
                    "preview": {"shape": [58, 9], "columns": [...], "memory_mb": 0.01}
                }
            }
        """
        from .artifact_accessors import df_schema, text_read
        from .artifact_store import artifact_info

        artifact_map = {}

        for event in tool_events:
            tool_name = event.get("tool_name")
            metadata = event.get("output", {}).get("metadata", {})

            # Find all artifact references (_aid suffix)
            for key, value in metadata.items():
                if not key.endswith("_aid"):
                    continue

                artifact_id = value
                artifact_name = key  # e.g., "global_metrics_aid"

                # Infer type from file extension (file-based artifact store)
                # Maps artifact_info().type_name to our internal type names
                try:
                    info = artifact_info(artifact_id)
                    type_name_map = {
                        "DataFrame": "dataframe",
                        "json": "json",
                        "str": "string",
                        "BytesIO": "plot",
                        "object": "unknown",
                    }
                    artifact_type = type_name_map.get(info.type_name, "unknown")
                except Exception:
                    artifact_type = "unknown"

                # Build base info
                preview_info = {
                    "type": artifact_type,
                    "source_tool": tool_name,
                    "artifact_name": artifact_name,
                }

                # Determine explorable status and tools based on type
                if artifact_type == "dataframe":
                    preview_info["explorable"] = True
                    preview_info["tools"] = ["df_head", "df_schema"]

                    # Get DataFrame preview
                    try:
                        schema = df_schema(artifact_id)
                        if isinstance(schema, dict):  # Success
                            preview_info["preview"] = {
                                "shape": schema["shape"],
                                "columns": schema["columns"],
                                "memory_mb": schema["memory_mb"]
                            }
                        else:  # Error string
                            preview_info["preview"] = {"error": schema}
                    except Exception as e:
                        preview_info["preview"] = {"error": str(e)}

                elif artifact_type == "json":
                    preview_info["explorable"] = True

                    # Check actual content type (dict vs list)
                    try:
                        obj = get_artifact(artifact_id)

                        if isinstance(obj, dict):
                            preview_info["tools"] = ["text_read", "get_metric", "list_metrics"]
                            keys = list(obj.keys())
                            preview_info["preview"] = {
                                "sample_keys": keys[:10],
                                "total_count": len(keys)
                            }
                        else:  # list or other
                            preview_info["tools"] = ["text_read"]
                            json_preview = text_read(artifact_id, max_chars=500)
                            preview_info["preview"] = json_preview
                    except Exception as e:
                        preview_info["tools"] = ["text_read"]
                        preview_info["preview"] = {"error": str(e)}

                elif artifact_type in {"markdown", "string", "text"}:
                    preview_info["explorable"] = True
                    preview_info["tools"] = ["text_read"]

                    # Get text preview
                    try:
                        text_preview = text_read(artifact_id, max_chars=500)
                        if not text_preview.startswith("Error"):  # Success
                            preview_info["preview"] = text_preview
                        else:  # Error string
                            preview_info["preview"] = {"error": text_preview}
                    except Exception as e:
                        preview_info["preview"] = {"error": str(e)}

                elif artifact_type in {"bytes", "figure", "plot"}:
                    preview_info["explorable"] = False
                    preview_info["tools"] = []
                    preview_info["preview"] = {"note": "Image/plot artifact (not text-explorable)"}

                else:
                    # Unknown type - show but don't suggest tools
                    preview_info["explorable"] = False
                    preview_info["tools"] = []
                    preview_info["preview"] = {"note": f"Unknown artifact type: {artifact_type}"}

                # Add to map (show ALL artifacts, don't filter)
                artifact_map[artifact_id] = preview_info

        if artifact_map:
            explorable_count = sum(1 for info in artifact_map.values() if info.get("explorable"))
            print(f"✓ Built artifact map with {len(artifact_map)} artifacts ({explorable_count} explorable)")
        return artifact_map

    def forward(
        self,
        user_prompt: str,
        data_paths: Dict[str, str],
        user_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete agent workflow.

        Args:
            user_prompt: User's question or request.
            data_paths: Data files, e.g. {"cgm": "path", "diet": "path"}.
            user_inputs: Participant metadata surfaced to the orchestrator for
                tool arguments (age, sex, bmi, ...).

        Returns:
            Memory dictionary with the complete workflow results.
        """
        # Scope LLM config to this agent run - prevents tools from clobbering global state
        with dspy.settings.context(lm=self.llm):
            return self._run(user_prompt, data_paths, user_inputs)

    def _run(
        self,
        user_prompt: str,
        data_paths: Dict[str, str],
        user_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full agent workflow."""
        from .data_paths import validate_data_paths, get_primary_path

        validate_data_paths(data_paths)

        print("📁 Data paths:")
        for data_type, path in data_paths.items():
            print(f"   {data_type}: {path}")

        # Initialize memory and timing
        memory = build_initial_memory()
        memory["user_prompt"] = user_prompt
        memory["data_paths"] = data_paths
        memory["csv_path"] = get_primary_path(data_paths)
        memory["context_docs"] = []
        memory["active_skills"] = self.skills
        timings: Dict[str, float] = {}

        # Generate skills context for injection at each stage
        orchestrator_skills = ""
        analyzer_skills = ""
        if self.skills:
            orchestrator_skills = self.skill_loader.format_skills_for_prompt(
                self.skills, injection_point="orchestrator"
            )
            analyzer_skills = self.skill_loader.format_skills_for_prompt(
                self.skills, injection_point="analyzer"
            )
            if orchestrator_skills:
                print("   Injecting skills into orchestrator")
            if analyzer_skills:
                print("   Injecting skills into analyzer")

        # Step 1: Orchestration - ReAct selects and executes tools
        t0 = time.time()
        plan = self.orchestrator(
            user_prompt=user_prompt,
            context_snippets=[],
            data_paths=data_paths,
            skills_context=orchestrator_skills,
            user_inputs=user_inputs,
        )
        timings["orchestrator_ms"] = round((time.time() - t0) * 1000, 2)
        memory["plan"] = plan

        events = plan.get("tool_events", [])
        print(f"✓ Using {len(events)} tool results from ReactOrchestrator")
        memory["tool_events"] = events

        # Step 2: Build the artifact preview map the analyzer explores from
        artifact_map = self._build_artifact_preview_map(events)
        memory["artifact_map"] = artifact_map

        # Step 3: Analysis - ReAct synthesises the report narrative
        t1 = time.time()
        analysis_text = self.analyzer(
            user_prompt=memory["user_prompt"],
            tool_events=memory["tool_events"],
            context_docs=memory["context_docs"],
            artifact_map=artifact_map,
            skills_context=analyzer_skills,
        )
        timings["analyzer_ms"] = round((time.time() - t1) * 1000, 2)
        memory["analysis_text"] = analysis_text
        memory["analyzer_iterations"] = {
            "used": getattr(self.analyzer, "last_iterations_used", 0),
            "max": self.analyzer.max_iters,
            "tool_calls": getattr(self.analyzer, "last_tool_calls", 0),
            "tool_calls_detail": getattr(self.analyzer, "last_tool_calls_detail", []),
        }

        # Step 4: Rendering - deterministic, no LLM
        t2 = time.time()
        report_path = self.reporter(
            analysis_text=analysis_text,
            tool_events=memory["tool_events"],
        )
        timings["reporter_ms"] = round((time.time() - t2) * 1000, 2)
        memory["final_report_path"] = report_path
        print(f"\n📄 Report written: {report_path}")

        # Step 5: Provenance tracking
        memory["provenance"]["timings"] = timings
        memory["provenance"]["tool_versions"] = {}
        for event in memory["tool_events"]:
            prov = event.get("output", {}).get("provenance", {})
            if isinstance(prov, dict):
                memory["provenance"]["tool_versions"][event["tool_name"]] = prov.get("tool_version")

        # dspy.LM.model is the LiteLLM model string
        memory["provenance"]["model_id"] = getattr(self.llm, "model", None)
        memory["provenance"]["temperature"] = getattr(self.llm, "temperature", 0.0)
        memory["provenance"]["iterations"] = {
            "orchestrator": {
                "used": plan.get("iterations_used", 0),
                "max": plan.get("max_iters", self.max_iters),
            },
            "analyzer": memory["analyzer_iterations"],
        }

        return memory
