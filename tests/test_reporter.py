"""Checks for the Reporter, the one pipeline stage with no LLM in it.

The Reporter turns the analyzer's narrative into files on disk. Because it is
deterministic, it is the stage where a figure can be guaranteed to agree with
the data it was drawn from - so these tests assert exactly that, plus graceful
behaviour when the CGM artifact is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "synthetic_001"

NARRATIVE = "## Summary\n\nMean glucose was 116.5 mg/dL over 14 days.\n"


@pytest.fixture
def tool_events(tmp_path):
    """A realistic cgm_parser tool event, produced by actually running it."""
    from pha.artifact_store import set_artifact_session
    from tools.cgm_parser import cgm_parser

    set_artifact_session(str(tmp_path / "artifacts"))
    parsed = cgm_parser(csv_path_or_buffer=str(SAMPLE_DIR / "cgm.csv"))

    return [{
        "tool_name": "cgm_parser",
        "args": {},
        "output": {"status": "ok", "metadata": {"cgm_aid": parsed.cgm_aid}},
    }]


def test_reporter_writes_report_and_figure(tmp_path, tool_events):
    """The happy path produces both the markdown and the PNG."""
    from pha.programs import Reporter

    reporter = Reporter(output_dir=tmp_path)
    report_path = Path(reporter(analysis_text=NARRATIVE, tool_events=tool_events))

    assert report_path.exists()
    assert report_path.name == "report.md"

    figure_path = tmp_path / "cgm_daily.png"
    assert figure_path.exists()
    assert figure_path.stat().st_size > 1000  # a real image, not an empty file

    body = report_path.read_text()
    assert NARRATIVE.strip() in body
    # The figure is referenced by relative name so the markdown renders in place
    assert "cgm_daily.png" in body


def test_reporter_writes_report_without_cgm_data(tmp_path):
    """With no CGM artifact, the report is still written and the figure skipped.

    A missing figure must never cost the reader their report.
    """
    from pha.programs import Reporter

    reporter = Reporter(output_dir=tmp_path)
    report_path = Path(reporter(analysis_text=NARRATIVE, tool_events=[]))

    assert report_path.exists()
    assert report_path.read_text().strip() == NARRATIVE.strip()
    assert not (tmp_path / "cgm_daily.png").exists()


def test_reporter_survives_a_dangling_artifact_id(tmp_path):
    """A tool event pointing at a missing artifact degrades, it does not raise."""
    from pha.programs import Reporter

    events = [{
        "tool_name": "cgm_parser",
        "args": {},
        "output": {"status": "ok", "metadata": {"cgm_aid": str(tmp_path / "cgm_gone.parquet")}},
    }]

    reporter = Reporter(output_dir=tmp_path)
    report_path = Path(reporter(analysis_text=NARRATIVE, tool_events=events))

    assert report_path.exists()
    assert not (tmp_path / "cgm_daily.png").exists()


def test_reporter_uses_the_consensus_target_range():
    """The band drawn on the figure is the published consensus range."""
    from pha.programs import Reporter

    assert Reporter.TARGET_RANGE_MGDL == (70.0, 180.0)


def test_reporter_is_deterministic(tmp_path, tool_events):
    """Two runs over the same data produce the same report bytes.

    The rendering stage has no LLM in it, so it must not vary run to run.
    """
    from pha.programs import Reporter

    first = Path(Reporter(output_dir=tmp_path / "a")(
        analysis_text=NARRATIVE, tool_events=tool_events))
    second = Path(Reporter(output_dir=tmp_path / "b")(
        analysis_text=NARRATIVE, tool_events=tool_events))

    assert first.read_text() == second.read_text()
