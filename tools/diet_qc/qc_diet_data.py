"""
Diet Logging Data QC Pipeline

Validates diet logging data using deterministic rule-based checks.
All rows are preserved - violations are flagged with severity levels.

The pipeline applies four rule groups (completeness, range, consistency,
integrity) and then rolls their flags up into a single per-row severity.
"""

import logging
import pandas as pd
from typing import Optional

from .rule_groups import (
    validate_completeness,
    validate_range_checks,
    validate_consistency,
    validate_data_integrity,
)
from .severity import calculate_severity
from .schemas import validate_required_columns, report_optional_columns
from .utils import prepare_dataframe_for_qc


def run_qc_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete QC pipeline with graceful column handling.

    Steps:
    1. Prepare the DataFrame (participant ID detection)
    2. Validate required columns, report missing optional columns
    3. Apply 4 rule groups (completeness, range, consistency, integrity)
    4. Roll flags up into an overall severity and failed-group list
    5. Return the enriched DataFrame (all input rows preserved)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with QC columns added, all input rows preserved

    Raises:
        ValueError: If required columns are missing
    """
    logging.info("=" * 60)
    logging.info("DIET LOGGING DATA QC PIPELINE")
    logging.info("=" * 60)

    # Step 1: Prepare DataFrame (handle participant ID detection)
    logging.info("[1/4] Preparing DataFrame...")
    df_prepared, participant_col, synthetic_used = prepare_dataframe_for_qc(df)

    # Step 2: Validate column structure
    logging.info("[2/4] Validating column structure...")
    is_valid, missing_required = validate_required_columns(df_prepared.columns.tolist())
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing_required}")

    present, missing = report_optional_columns(df_prepared.columns.tolist())
    if missing:
        logging.info(f"Optional columns missing: {len(missing)}")
        for col in missing[:5]:
            logging.info(f"  - {col}")
        if len(missing) > 5:
            logging.info(f"  ... and {len(missing) - 5} more")

    logging.info(f"{len(df_prepared)} rows, {len(df_prepared.columns)} columns")

    result_df = df_prepared.copy()

    # Step 3: Apply rule groups
    logging.info("[3/4] Applying rule-based QC...")
    result_df = validate_completeness(result_df)
    result_df = validate_range_checks(result_df)
    result_df = validate_consistency(result_df)
    result_df = validate_data_integrity(result_df)

    # Step 4: Roll up severity
    logging.info("[4/4] Calculating severity...")
    result_df = calculate_severity(result_df)

    # Drop the synthetic participant ID if we created one
    if synthetic_used and '_participant_id' in result_df.columns:
        result_df = result_df.drop(columns=['_participant_id'])

    # Summary
    severity_counts = result_df['qc_severity'].value_counts()
    logging.info("Severity distribution:")
    for severity in ['fail', 'check', 'pass']:
        count = severity_counts.get(severity, 0)
        pct = (count / len(result_df) * 100) if len(result_df) > 0 else 0
        logging.info(f"  - {severity}: {count} ({pct:.1f}%)")

    logging.info(f"Pipeline complete: {len(result_df)} rows (all input rows preserved)")

    return result_df
