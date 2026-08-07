"""Every shipped module imports, and every CLI entry point starts.

This exists because a docstring edit once left `evals/core/runner.py` with an
unterminated string literal, and the rest of the suite did not notice: nothing
else imports the runner, so a syntax error in it was invisible until someone
ran the CLI by hand.

These tests are cheap and catch a whole class of packaging and edit mistakes
that unit tests aimed at behaviour will miss.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ["pha", "tools", "evals", "tests"]

# Modules that should import cleanly with no API key and no network
IMPORTABLE = [
    "pha.artifact_accessors",
    "pha.artifact_store",
    "pha.data_paths",
    "pha.llm",
    "pha.memory",
    "pha.programs",
    "pha.skill_loader",
    "pha.tool_decorator",
    "pha.trace_utils",
    "pha.utils",
    "pha.validators",
    "tools.cgm_metrics",
    "tools.cgm_parser",
    "tools.diet_qc",
    "tools.meal_parser",
    "evals.core.eval_decorator",
    "evals.core.registry",
    "evals.core.runner",
    "evals.core.utils",
    "evals.evaluators.clinical_language",
    "evals.evaluators.completeness",
    "evals.evaluators.groundedness",
    "evals.evaluators.numerical",
]


def _python_files() -> list[Path]:
    files = []
    for package in PACKAGES:
        files.extend((REPO_ROOT / package).rglob("*.py"))
    files.extend(REPO_ROOT.glob("*.py"))
    files.extend((REPO_ROOT / "data").glob("*.py"))
    return sorted(set(files))


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_file_parses(path: Path):
    """Every Python file in the repository is syntactically valid."""
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("module", IMPORTABLE)
def test_module_imports(module: str):
    """Every shipped module imports without credentials or network."""
    __import__(module)


@pytest.mark.parametrize(
    "argv",
    [
        ["run_report.py", "--help"],
        ["-m", "evals.core.runner", "--help"],
        ["-m", "evals.core.runner", "list", "--no-llm"],
        ["data/make_synthetic_data.py", "--help"],
    ],
    ids=["run_report", "runner_help", "runner_list", "make_data"],
)
def test_cli_entry_point_starts(argv: list[str]):
    """Each CLI entry point runs without erroring."""
    result = subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{' '.join(argv)} exited {result.returncode}\n{result.stderr[-2000:]}"
    )


def test_runner_list_finds_every_evaluator():
    """`runner list` reports the same evals the registry discovered.

    The registry swallows a failed evaluator import into a printed warning, so
    a broken evaluator silently disappears from the suite while the runner
    still reports success over the survivors. This asserts the two agree.
    """
    from evals.core.registry import list_evals

    discovered = {e.name for e in list_evals()}
    assert discovered, "registry discovered no evaluators"

    result = subprocess.run(
        [sys.executable, "-m", "evals.core.runner", "list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    missing = sorted(name for name in discovered if name not in result.stdout)
    assert not missing, f"runner list omitted: {missing}"
    assert "Warning: Failed to import" not in result.stdout + result.stderr
