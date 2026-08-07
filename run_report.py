#!/usr/bin/env python
"""Generate a health report from CGM and diet-log data.

Quickstart:

    export ANTHROPIC_API_KEY=...
    python run_report.py

That runs the bundled synthetic participant with both skills active and writes
the report, the figure and the execution trace to out/.

Other uses:

    # A different participant directory (needs cgm.csv, optionally diet.csv
    # and metadata.json)
    python run_report.py --sample data/synthetic_001 --out out/

    # Ablate the skills layer, which is how the skill contribution is measured
    python run_report.py --skills                       # none
    python run_report.py --skills clinical-language-guard

    # Recompute ground truth for the eval harness (no LLM involved)
    python run_report.py --write-ground-truth

Then score the report:

    python -m evals.core.runner run --report out/report.md \\
        --ground-truth data/synthetic_001/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SAMPLE = REPO_ROOT / "data" / "synthetic_001"
DEFAULT_SKILLS = ["clinical-language-guard", "diet-qc-guard", "metabolic_health_report"]

# Mirrors the default full-report prompt used to generate the p0 sample reports
# in the internal system, minus the requests this build cannot ground: it has no
# population percentiles, no digital-twin comparison, and no PPGR-based meal
# optimisation, so asking for them would only invite fabrication.
# The paper's three matrix prompts are the only prompts this repo ships. The
# default is the one whose asks this four-tool build can answer furthest; the
# others are equally valid inputs, they just produce more declines.
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts" / "metabolic_scorecard.md"


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    """Read the user prompt from a file.

    Args:
        prompt_file: Prompt to use. Defaults to the metabolic scorecard.

    Returns:
        The prompt text.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = prompt_file or DEFAULT_PROMPT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


def load_sample(sample_dir: Path) -> tuple[Dict[str, str], Dict]:
    """Read a participant directory into data paths and metadata.

    Args:
        sample_dir: Directory holding cgm.csv, and optionally diet.csv and
            metadata.json.

    Returns:
        Tuple of (data_paths, metadata).

    Raises:
        FileNotFoundError: If the directory or its cgm.csv is missing.
    """
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    cgm_path = sample_dir / "cgm.csv"
    if not cgm_path.exists():
        raise FileNotFoundError(f"No cgm.csv in {sample_dir}")

    data_paths = {"cgm": str(cgm_path)}

    diet_path = sample_dir / "diet.csv"
    if diet_path.exists():
        data_paths["diet"] = str(diet_path)
    else:
        print(f"Note: no diet.csv in {sample_dir}, dietary sections will be omitted")

    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return data_paths, metadata


def write_ground_truth(sample_dir: Path) -> Path:
    """Compute ground-truth CGM metrics for a sample, without an LLM.

    Runs cgm_parser and cgm_metrics directly and writes the resulting metric
    dict to ground_truth.json. This is what eval_numerical scores the report
    against: the eval checks that the LLM transcribed tool output faithfully,
    not that the metrics themselves are correct.

    Args:
        sample_dir: Participant directory containing cgm.csv.

    Returns:
        Path to the written ground_truth.json.
    """
    from pha.artifact_store import get_artifact, set_artifact_session
    from tools.cgm_parser import cgm_parser
    from tools.cgm_metrics import cgm_metrics

    set_artifact_session(str(sample_dir / "artifacts"))

    parsed = cgm_parser(csv_path_or_buffer=str(sample_dir / "cgm.csv"))
    metrics = cgm_metrics(cgm_aid=parsed.cgm_aid)
    full_metrics = get_artifact(metrics.global_metrics_aid)

    # Keep only scalar metrics; cgm_episodes is a nested list
    ground_truth = {
        k: v for k, v in full_metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    out_path = sample_dir / "ground_truth.json"
    out_path.write_text(json.dumps(ground_truth, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path} ({len(ground_truth)} metrics)")
    return out_path


def generate_report(
    sample_dir: Path,
    out_dir: Path,
    skills: Optional[List[str]] = None,
    max_iters: int = 10,
    prompt_file: Optional[Path] = None,
) -> str:
    """Run the agent over a sample and write the report.

    Args:
        sample_dir: Participant directory.
        out_dir: Directory for report.md, cgm_daily.png, trace.json, artifacts/.
        skills: Skill names to activate. Empty list means no skills.
        max_iters: Maximum ReactOrchestrator iterations.
        prompt_file: Prompt to use. Defaults to the metabolic scorecard.

    Returns:
        The generated report markdown.
    """
    from pha.artifact_store import set_artifact_session
    from pha.llm import get_lm
    from pha.programs import Agent
    from pha.trace_utils import save_trace
    from tools.cgm_parser import cgm_parser
    from tools.cgm_metrics import cgm_metrics
    from tools.meal_parser import meal_parser
    from tools.diet_qc import qc_diet_data

    data_paths, metadata = load_sample(sample_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    set_artifact_session(str(out_dir / "artifacts"))

    # Only offer the diet tools when there is a diet log to parse
    tools = [cgm_parser, cgm_metrics]
    if "diet" in data_paths:
        tools += [meal_parser, qc_diet_data]

    lm = get_lm(tier="HIGH")

    agent = Agent(
        llm=lm,
        tools=tools,
        output_dir=out_dir,
        max_iters=max_iters,
        skills=skills if skills is not None else DEFAULT_SKILLS,
    )

    memory = agent(
        user_prompt=load_prompt(prompt_file),
        data_paths=data_paths,
        user_inputs=metadata,
    )

    save_trace(memory, lm, out_dir)
    return memory.get("analysis_text", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample", type=Path, default=DEFAULT_SAMPLE,
        help=f"Participant directory (default: {DEFAULT_SAMPLE.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "out",
        help="Output directory (default: out/)",
    )
    parser.add_argument(
        "--skills", nargs="*", default=None, metavar="NAME",
        help=f"Skills to activate (default: {' '.join(DEFAULT_SKILLS)}). "
             "Pass with no names to disable all skills.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=10,
        help="Maximum orchestrator iterations (default: 10)",
    )
    parser.add_argument(
        "--prompt", type=Path, default=None, metavar="FILE",
        help="Prompt file (default: prompts/metabolic_scorecard.md). prompts/ "
             "holds all three prompts from the paper's report matrix.",
    )
    parser.add_argument(
        "--write-ground-truth", action="store_true",
        help="Recompute ground_truth.json for the sample and exit (no LLM used)",
    )
    args = parser.parse_args()

    if args.write_ground_truth:
        write_ground_truth(args.sample)
        return

    report = generate_report(
        sample_dir=args.sample,
        out_dir=args.out,
        skills=args.skills,
        max_iters=args.max_iters,
        prompt_file=args.prompt,
    )

    if not report.strip():
        print("\n⚠️  The agent returned an empty report. Check the trace in "
              f"{args.out} for what happened.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Report:   {args.out / 'report.md'}")
    print(f"Figure:   {args.out / 'cgm_daily.png'}")
    print(f"Trace:    {args.out}/trace_*.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
