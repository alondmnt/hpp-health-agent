"""Checks for the deterministic evaluators, against fixture reports.

No LLM and no network: the groundedness evaluator is LLM-backed and is
exercised by the end-to-end run instead, not here.

The fixtures below are written by hand rather than generated, so that a change
in eval behaviour shows up as a test failure rather than as a quietly different
score on a freshly generated report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO_ROOT / "data" / "synthetic_001" / "ground_truth.json"

# A report that faithfully transcribes what the tools computed. It covers every
# metric eval_numerical thresholds, so a partial report cannot masquerade as a
# perfect score.
FAITHFUL_REPORT = """# Metabolic Health Report

## Glycaemic Control

| Metric | Value |
| --- | --- |
| **Mean Glucose** | {cgm_mean:.2f} mg/dL |
| **CV** | {cgm_cv:.2f}% |
| Time in range (70-180 mg/dL) | {cgm_in_range_70_180:.2f}% |
| **eA1c** | {cgm_ea1c:.2f}% |
| **GMI** | {cgm_gmi:.2f}% |
| **ADRR** | {cgm_adrr:.2f} |
| **LBGI** | {cgm_lbgi:.2f} |
| **HBGI** | {cgm_hbgi:.2f} |

Your readings sat within the consensus target range for most of the recording.
"""

# The same report with the mean glucose invented rather than transcribed
FABRICATED_REPORT = FAITHFUL_REPORT.replace("{cgm_mean:.2f}", "98.00")


@pytest.fixture(scope="module")
def ground_truth():
    if not GROUND_TRUTH.exists():
        pytest.skip(
            f"{GROUND_TRUTH} missing; run: python run_report.py --write-ground-truth"
        )
    return json.loads(GROUND_TRUTH.read_text())


@pytest.fixture(scope="module")
def numerical_config():
    import yaml

    config = yaml.safe_load((REPO_ROOT / "evals" / "eval_config.yaml").read_text())
    return config["eval_numerical"]


def test_numerical_passes_a_faithful_report(ground_truth, numerical_config):
    """A report that transcribes tool output exactly should pass."""
    from evals.evaluators.numerical import eval_numerical

    report = FAITHFUL_REPORT.format(**ground_truth)
    result = eval_numerical(report, ground_truth, config=numerical_config)

    assert result.pass_, f"faithful report failed: {result.failures}"
    assert result.score == pytest.approx(1.0)


def test_numerical_catches_a_fabricated_metric(ground_truth, numerical_config):
    """A single invented number must fail the eval.

    This is the whole point of the harness: the fabricated report reads exactly
    as well as the faithful one, and only the comparison against tool output
    tells them apart.
    """
    from evals.evaluators.numerical import eval_numerical

    report = FABRICATED_REPORT.format(**{k: v for k, v in ground_truth.items()})
    result = eval_numerical(report, ground_truth, config=numerical_config)

    assert not result.pass_
    assert result.score < 1.0
    assert any("cgm_mean" in f for f in result.failures)


def test_numerical_score_never_contradicts_the_verdict(ground_truth, numerical_config):
    """A passing report cannot score 0, and a failing one cannot score 1."""
    from evals.evaluators.numerical import eval_numerical

    for report in (FAITHFUL_REPORT.format(**ground_truth),
                   FABRICATED_REPORT.format(**ground_truth)):
        result = eval_numerical(report, ground_truth, config=numerical_config)
        if result.pass_:
            assert result.score > 0.0
        else:
            assert result.score < 1.0


def test_completeness_requires_the_mandatory_metrics():
    """The three mandatory metrics must be present with values."""
    from evals.evaluators.completeness import eval_critical_cgm_metrics

    complete = "Mean glucose: 116.5 mg/dL\nCV: 30.6%\nTime in range 70-180: 93.5%\n"
    assert eval_critical_cgm_metrics(complete).pass_

    incomplete = "Mean glucose: 116.5 mg/dL\n"
    result = eval_critical_cgm_metrics(incomplete)
    assert not result.pass_
    assert result.score < 1.0


def test_clinical_language_flags_diagnostic_terms():
    """Diagnostic and alarmist language is caught; neutral phrasing is not."""
    from evals.evaluators.clinical_language import eval_clinical_language

    neutral = (
        "Your mean glucose was 116.5 mg/dL and time in range was 93.5%. "
        "These may be worth discussing with your healthcare provider."
    )
    assert eval_clinical_language(neutral).pass_

    alarmist = (
        "Your readings show a dangerous pattern consistent with pre-diabetic "
        "status. This is deeply concerning and requires IMMEDIATE ATTENTION."
    )
    result = eval_clinical_language(alarmist)
    assert not result.pass_
    assert result.failures


def test_every_registered_eval_accepts_the_standard_signature():
    """Each evaluator takes (report_text, ground_truth, config=...).

    The runner calls every eval this way, and a mismatch surfaces as a caught
    exception reported as a failure rather than as a clear error.
    """
    import inspect

    from evals.core.registry import list_evals

    evals = list_evals()
    assert evals, "no evaluators discovered"

    for eval_info in evals:
        params = inspect.signature(eval_info.func).parameters
        assert "report_text" in params, f"{eval_info.name} lacks report_text"
        assert "ground_truth" in params, f"{eval_info.name} lacks ground_truth"
        assert "config" in params, f"{eval_info.name} lacks config"


def test_eval_config_covers_only_shipped_evals():
    """No stale thresholds for evaluators that are not shipped."""
    import yaml

    from evals.core.registry import list_evals

    config = yaml.safe_load((REPO_ROOT / "evals" / "eval_config.yaml").read_text())
    shipped = {e.name for e in list_evals()}

    stale = sorted(set(config) - shipped)
    assert not stale, f"eval_config.yaml configures evals that do not exist: {stale}"
