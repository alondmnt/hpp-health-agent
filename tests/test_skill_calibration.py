"""Skill-tool calibration: every identifier a skill names must actually exist.

This is the mechanical form of the finding described in the paper: a skill that
references a metric no tool computes teaches the model to fabricate it, and the
fabrication is invisible in the output. Rather than relying on review to catch
that, this test extracts every backticked identifier from every shipped skill
and asserts it resolves to one of:

  - a shipped tool name
  - a bounded artifact accessor
  - a field on some tool's input or output model
  - a standardized metric name in the CGM metrics catalog
  - a column in the bundled sample data

Prose in a skill is not checked, only backticked identifiers. That is
deliberate: the FORBIDDEN sections name things that do NOT exist, and they name
them in prose precisely so this test does not trip on them.

If this test fails, either the skill is wrong or the tool changed under it.
Do not add the identifier to an allowlist without checking which.
"""
from __future__ import annotations

import csv
import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
METRICS_CATALOG = REPO_ROOT / "tools" / "cgm_metrics" / "config" / "cgm_metrics_catalog.csv"
SAMPLE_DIR = REPO_ROOT / "data" / "synthetic_001"

# Bounded accessors the analyzer may call (pha/artifact_accessors.py)
ACCESSORS = {"get_metric", "list_metrics", "df_head", "df_schema", "text_read"}

# Identifiers that are English words or generic placeholders rather than
# references to anything in the codebase.
IGNORED = {"name", "n", "true", "false", "none"}

# Matches a backticked lower_snake_case token, with or without a call
IDENTIFIER_RE = re.compile(r"`([a-z_][a-z0-9_]*)\s*(?:\(|`)")


def _metadata_fields(tool_func) -> set[str]:
    """Field names of a tool's output metadata model."""
    annotation = tool_func._output_model.model_fields["metadata"].annotation
    for candidate in typing.get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return set(candidate.model_fields)
    raise AssertionError(f"Could not resolve metadata model for {tool_func}")


def _shipped_tools():
    """Import the four shipped tools."""
    from tools.cgm_parser import cgm_parser
    from tools.cgm_metrics import cgm_metrics
    from tools.meal_parser import meal_parser
    from tools.diet_qc import qc_diet_data

    return [cgm_parser, cgm_metrics, meal_parser, qc_diet_data]


def _known_identifiers() -> set[str]:
    """Build the set of identifiers a skill is allowed to reference."""
    known: set[str] = set(ACCESSORS) | set(IGNORED)

    for tool_func in _shipped_tools():
        known.add(tool_func.__tool_name__)
        # Tool arguments
        known.update(tool_func._input_model.model_fields)
        # Metadata fields the tool returns. The envelope types metadata as
        # Optional[TheModel], so unwrap the union before reading its fields.
        known.update(_metadata_fields(tool_func))

    # The 55 standardized metric names available inside global_metrics_aid
    with open(METRICS_CATALOG, newline="") as f:
        known.update(row["standardized_name"] for row in csv.DictReader(f))

    # Columns of the bundled sample data, which the analyzer reads via df_head
    for csv_name in ("cgm.csv", "diet.csv"):
        path = SAMPLE_DIR / csv_name
        if path.exists():
            with open(path, newline="") as f:
                known.update(next(csv.reader(f)))

    return known


def _skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*.md"))


def test_skills_exist():
    """There is at least one skill to check."""
    assert _skill_files(), f"No skill files found in {SKILLS_DIR}"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.stem)
def test_skill_identifiers_resolve(skill_path: Path):
    """Every backticked identifier in a skill resolves to something real."""
    known = _known_identifiers()
    referenced = set(IDENTIFIER_RE.findall(skill_path.read_text()))

    unresolved = sorted(referenced - known)
    assert not unresolved, (
        f"{skill_path.name} references identifiers that no shipped tool provides: "
        f"{unresolved}. Either the skill is wrong, or a tool changed under it. "
        "An unresolved metric name is how a skill teaches the model to fabricate."
    )


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.stem)
def test_skill_declared_tools_are_shipped(skill_path: Path):
    """The `tools:` frontmatter list, where present, names only shipped tools."""
    from pha.skill_loader import SkillLoader

    loader = SkillLoader(SKILLS_DIR)
    skill = loader._load_skill(skill_path.stem)

    shipped = {t.__tool_name__ for t in _shipped_tools()}
    declared = set(skill.tools)

    missing = sorted(declared - shipped)
    assert not missing, (
        f"{skill_path.name} declares tools that are not shipped: {missing}"
    )


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.stem)
def test_skill_injection_points_are_valid(skill_path: Path):
    """Injection points resolve to stages the agent actually injects at."""
    from pha.skill_loader import SkillLoader

    loader = SkillLoader(SKILLS_DIR)
    skill = loader._load_skill(skill_path.stem)

    assert skill.injection_points, f"{skill_path.name} has no injection points"
    assert set(skill.injection_points) <= {"orchestrator", "analyzer"}
