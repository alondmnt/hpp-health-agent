"""End-to-end checks for the four tools, over the bundled synthetic data.

No LLM and no network. These run the real tool chain against the real sample
files, so they also serve as a check that the bundled data still parses.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "synthetic_001"


@pytest.fixture(scope="module")
def artifact_session(tmp_path_factory):
    """Point the artifact store at a temporary directory for these tests."""
    from pha.artifact_store import set_artifact_session

    session = tmp_path_factory.mktemp("artifacts")
    set_artifact_session(str(session))
    return session


@pytest.fixture(scope="module")
def parsed_cgm(artifact_session):
    from tools.cgm_parser import cgm_parser

    return cgm_parser(csv_path_or_buffer=str(SAMPLE_DIR / "cgm.csv"))


@pytest.fixture(scope="module")
def parsed_meals(artifact_session):
    from tools.meal_parser import meal_parser

    return meal_parser(file_path=str(SAMPLE_DIR / "diet.csv"))


def test_cgm_parser_reads_sample(parsed_cgm):
    """cgm_parser handles the bundled CGM file with default column names."""
    assert parsed_cgm.n_days == 14
    assert parsed_cgm.n_rows == 14 * 24 * 4  # 15-minute cadence
    assert parsed_cgm.sampling_interval_seconds == pytest.approx(900.0)
    assert parsed_cgm.nan_count == 0
    assert parsed_cgm.gap_count == 0
    assert parsed_cgm.original_units == "mg/dL"
    assert parsed_cgm.cgm_aid


def test_cgm_parser_artifact_round_trips(parsed_cgm):
    """The stored artifact loads back as a non-empty DataFrame."""
    import pandas as pd
    from pha.artifact_store import get_artifact

    df = get_artifact(parsed_cgm.cgm_aid, expected_prefix="cgm")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == parsed_cgm.n_rows


def test_cgm_parser_converts_mmol(artifact_session):
    """mmol/L input is converted to mg/dL by the standard factor."""
    from tools.cgm_parser import cgm_parser

    csv_data = "collection_timestamp,glucose\n2024-01-01 00:00,5.5\n2024-01-01 00:15,6.0\n"
    result = cgm_parser(csv_string=csv_data, units="mmol/L")
    assert result.conversion_factor == pytest.approx(18.0)
    # original_units records what came in, not what was stored
    assert result.original_units == "mmol/L"


def test_cgm_metrics_populates_every_declared_field(parsed_cgm):
    """Every field cgm_metrics declares is present and finite."""
    from tools.cgm_metrics import cgm_metrics

    result = cgm_metrics(cgm_aid=parsed_cgm.cgm_aid)

    assert result.days_total == 14
    assert result.days_valid > 0
    assert result.units == "mg/dL"

    for field in ("cgm_mean", "cgm_cv", "cgm_sd", "cgm_gmi",
                  "cgm_in_range_70_180", "cgm_mage", "cgm_j_index"):
        value = getattr(result, field)
        assert isinstance(value, float), f"{field} is not a float"
        assert math.isfinite(value), f"{field} is not finite"

    # Percentages are on a 0-100 scale
    assert 0.0 <= result.cgm_in_range_70_180 <= 100.0
    assert 0.0 <= result.cgm_below_70 <= 100.0

    for aid_field in ("global_metrics_aid", "metric_units_aid", "daily_metrics_aid"):
        assert getattr(result, aid_field), f"{aid_field} is empty"


def test_cgm_metrics_artifacts_are_readable(parsed_cgm):
    """The bounded accessors work against the metrics artifacts."""
    from pha.artifact_accessors import df_head, df_schema, get_metric, list_metrics
    from tools.cgm_metrics import cgm_metrics

    result = cgm_metrics(cgm_aid=parsed_cgm.cgm_aid)

    names = list_metrics(result.global_metrics_aid, limit=100)
    assert isinstance(names, list) and len(names) > 40

    mean = get_metric(result.global_metrics_aid, "cgm_mean")
    assert isinstance(mean, float)
    assert mean == pytest.approx(result.cgm_mean)

    schema = df_schema(result.daily_metrics_aid)
    assert isinstance(schema, dict)
    assert schema["shape"][0] == 14  # one row per day

    head = df_head(result.daily_metrics_aid, n=5)
    assert isinstance(head, dict) and "date" in head


def test_df_head_enforces_its_bound(parsed_cgm):
    """df_head refuses n > 100 rather than flooding the context."""
    from pha.artifact_accessors import df_head

    out = df_head(parsed_cgm.cgm_aid, n=500)
    assert isinstance(out, str) and out.startswith("Error")


def test_meal_parser_reads_sample(parsed_meals):
    """meal_parser handles the bundled diet log."""
    assert parsed_meals.rows_parsed == 126  # 14 days x 3 meals x 3 items
    assert parsed_meals.date_range == "2024-01-01 to 2024-01-14"
    assert "calories_kcal" in parsed_meals.columns
    assert parsed_meals.meal_aid


def test_meal_parser_rejects_missing_columns(artifact_session, tmp_path):
    """A file without the required columns fails loudly, not silently."""
    from tools.meal_parser import meal_parser

    bad = tmp_path / "bad.csv"
    bad.write_text("collection_timestamp,product_name\n2024-01-01 08:00,Toast\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        meal_parser(file_path=str(bad))


def test_diet_qc_flags_the_seeded_faults(parsed_meals):
    """QC finds exactly the faults make_synthetic_data.py seeds, and no others.

    The generator plants one completeness fault, one range fault and one
    consistency fault. If this count drifts, either a rule changed or the
    synthetic food table stopped being self-consistent under the Atwater
    factors - both worth knowing.
    """
    from tools.diet_qc import qc_diet_data

    result = qc_diet_data(meal_aid=parsed_meals.meal_aid)

    assert result.total_rows == 126
    assert result.total_days == 14
    assert result.total_participants == 1
    assert result.severity_counts.get("fail", 0) == 3

    groups = " ".join(result.failed_groups)
    assert "completeness" in groups
    assert "range" in groups
    assert "consistency" in groups


def test_diet_qc_preserves_every_row(parsed_meals):
    """QC flags rows, it never drops or corrects them."""
    from pha.artifact_store import get_artifact
    from tools.diet_qc import qc_diet_data

    result = qc_diet_data(meal_aid=parsed_meals.meal_aid)
    qc_df = get_artifact(result.result_aid)

    assert len(qc_df) == parsed_meals.rows_parsed
    for col in ("qc_completeness", "qc_range", "qc_consistency",
                "qc_integrity", "qc_severity", "qc_failed_groups"):
        assert col in qc_df.columns

    # The synthetic participant column is stripped from the output
    assert "_participant_id" not in qc_df.columns


def test_tool_errors_return_envelopes_not_exceptions(artifact_session):
    """The @tool decorator turns failures into structured envelopes.

    The ReAct loop depends on this: a tool that raises would abort the run,
    where an error envelope lets the agent read the problem and adapt.
    """
    from tools.cgm_parser import cgm_parser

    envelope = cgm_parser.invoke({"csv_path_or_buffer": "does/not/exist.csv"})
    assert envelope.status == "error"
    assert envelope.error is not None
    assert envelope.error.code
