"""
Diet Data QC - Validate diet logging data quality

## Purpose
Deterministic QC pipeline. Applies 4 rule groups (completeness, range,
consistency, integrity) and rolls their flags up into a per-row severity.
All input rows are preserved; nothing is dropped or corrected.

## When to Use
- Validating diet logging datasets before analysis
- Quality assurance for nutritional data
- Detecting data entry errors or implausible values

## When NOT to Use
- For filling missing data or fixing errors in diet logging (use augmentation tools)
- For segmenting data into meals (use segmentation tools)

## Requirements
- Input: DataFrame with required column local_timestamp
- Participant ID: participant_uuid or participant_id (optional - a synthetic ID
  is used when absent, and stripped from the output)
- Optional columns: calories_kcal, carbohydrate_g, lipid_g, protein_g, etc.

## Examples
    # Method 1: Chained after meal_parser (the usual path)
    parsed = meal_parser(file_path="data/synthetic_001/diet.csv")
    result = qc_diet_data(meal_aid=parsed.meal_aid)
    qc_df = get_artifact(result.result_aid)

    # Method 2: Straight from a CSV
    result = qc_diet_data(file_path="data/synthetic_001/diet.csv")
    qc_df = get_artifact(result.result_aid)
"""
from __future__ import annotations

from typing import Annotated, Optional
from pydantic import BaseModel, Field
import pandas as pd

from pha.tool_decorator import tool
from pha.artifact_store import get_artifact, put_artifact

from .qc_diet_data import run_qc_pipeline
from .schemas import report_optional_columns


# Artifact prefix constants
_EXPECTED_INPUT_PREFIX = "meal"
_OUTPUT_PREFIX = "dietQC"


class QCDietDataMetadata(BaseModel):
    """Output metadata for the qc_diet_data tool.

    This metadata object carries only aggregate statistics, so it is safe to
    surface to an LLM. Note that result_aid points at a DataFrame containing
    every input row, participant identifiers included - the artifact is not
    de-identified, only this envelope is.
    """
    result_aid: str = Field(description="Artifact store ID for QC-validated DataFrame")
    total_participants: int = Field(description="Number of unique participants in dataset", ge=0)
    total_rows: int = Field(description="Total rows processed", ge=0)
    total_days: int = Field(description="Number of unique logging days", ge=0)
    severity_counts: dict = Field(
        description="Distribution: {pass: X, check: Y, fail: Z}",
        default_factory=dict
    )
    failed_groups: dict = Field(
        description="Top 5 failed rule group combinations",
        default_factory=dict
    )
    optional_columns_missing: int = Field(
        description="Number of optional columns missing",
        ge=0
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered"
    )


@tool(
    version="1.0.0",
    categories=["DataQuality", "Diet"],
    description="Validate diet logging data with deterministic rule-based checks"
)
def qc_diet_data(
    meal_aid: Annotated[Optional[str], Field(
        description="Artifact ID of meal DataFrame (from meal_parser)",
        min_length=1
    )] = None,
    *,
    file_path: Annotated[Optional[str], Field(
        description="Path to CSV file (alternative to meal_aid)"
    )] = None,
) -> QCDietDataMetadata:
    """Validate diet logging data using deterministic rule-based checks.

    Applies 4 rule groups (completeness, range, consistency, integrity) and
    rolls their flags up into a per-row severity of pass, check, or fail.
    All input rows are preserved with QC flag columns appended.

    Input Data Requirements:
        Required: local_timestamp
        Optional but recommended: participant_uuid or participant_id
            (a synthetic ID is used when absent, and stripped from the output)
        Optional nutrients: calories_kcal, carbohydrate_g, protein_g, lipid_g, etc.

    Args:
        meal_aid: Artifact ID of meal DataFrame (from meal_parser). Use this OR file_path.
        file_path: Direct CSV path. Use this OR meal_aid.

    Returns:
        QCDietDataMetadata with validation results
        
    Raises:
        ValueError: Invalid input, missing columns, or both/neither inputs
        KeyError: Artifact not found
        FileNotFoundError: File not found
    """
    from pathlib import Path
    
    warnings = []
    
    # Validate input method
    if meal_aid and file_path:
        raise ValueError("Cannot provide both meal_aid and file_path.")
    if not meal_aid and not file_path:
        raise ValueError("Must provide either meal_aid or file_path.")
    
    # Load data
    if file_path:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path_obj.suffix.lower() != '.csv':
            raise ValueError(f"Unsupported format: {file_path_obj.suffix}. Use .csv")
        df = pd.read_csv(file_path_obj)
    else:
        try:
            df = get_artifact(meal_aid, expected_prefix=_EXPECTED_INPUT_PREFIX)
        except KeyError:
            raise KeyError(f"Artifact '{meal_aid}' not found")
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Invalid type: {type(df)}. Expected DataFrame.")
    
    # Run QC pipeline
    try:
        result_df = run_qc_pipeline(df)
    except Exception as e:
        raise ValueError(f"QC pipeline failed: {str(e)}")
    
    # Store result
    result_aid = put_artifact(result_df, prefix=_OUTPUT_PREFIX)
    
    # Build metadata with aggregate statistics (privacy-focused)
    severity_counts = result_df['qc_severity'].value_counts().to_dict()
    
    # Calculate total_participants (count unique values in participant column)
    from .utils import get_participant_col
    participant_col = get_participant_col(result_df)
    if participant_col:
        total_participants = result_df[participant_col].nunique()
    else:
        # No participant column exists (shouldn't happen after prepare_dataframe_for_qc)
        total_participants = 1
    
    # Calculate total_days (unique dates from local_timestamp)
    if 'local_timestamp' in result_df.columns:
        result_df_temp = result_df.copy()
        result_df_temp['_date'] = pd.to_datetime(result_df_temp['local_timestamp']).dt.date
        total_days = result_df_temp['_date'].nunique()
    else:
        total_days = 0
    
    failed_groups = {}
    if 'qc_failed_groups' in result_df.columns:
        failed_groups = (
            result_df[result_df['qc_failed_groups'] != '']['qc_failed_groups']
            .value_counts()
            .head(5)
            .to_dict()
        )
    
    _, missing = report_optional_columns(df.columns.tolist())
    
    return QCDietDataMetadata(
        result_aid=result_aid,
        total_participants=total_participants,
        total_rows=len(result_df),
        total_days=total_days,
        severity_counts=severity_counts,
        failed_groups=failed_groups,
        optional_columns_missing=len(missing),
        warnings=warnings
    )

