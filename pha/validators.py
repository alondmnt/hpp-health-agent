"""CGM data validation and profiling utilities.

This module provides validation and profiling functions for CGM time series data,
including data quality checks, gap detection, and statistical profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CGMProfile:
    """Profile summary of a CGM time series dataset.
    
    Attributes:
        n_rows: Total number of data rows.
        n_days: Number of unique days in the dataset.
        start_iso: ISO format timestamp of first reading.
        end_iso: ISO format timestamp of last reading.
        cadence_seconds_est: Estimated recording interval in seconds (median).
        cadence_iqr_seconds: IQR of recording intervals in seconds.
        per_day_counts: DataFrame with per-day point counts.
        gap_stats: Dictionary with gap statistics (num_gaps, max_gap_min, total_gap_min).
        nan_count: Number of missing/NaN glucose values.
        non_numeric_rows: Number of rows with non-numeric glucose values.
        duplicates_dropped: Number of duplicate timestamp entries removed.
    """
    n_rows: int
    n_days: int
    start_iso: Optional[str]
    end_iso: Optional[str]
    cadence_seconds_est: Optional[float]
    cadence_iqr_seconds: Optional[float]
    per_day_counts: pd.DataFrame
    gap_stats: Dict[str, float]
    nan_count: int
    non_numeric_rows: int
    duplicates_dropped: int


def validate_cgm_dataframe(df: pd.DataFrame, timestamp_col: str, glucose_col: str) -> None:
    """Validate that required columns exist in a CGM DataFrame.
    
    Args:
        df: The DataFrame to validate.
        timestamp_col: Name of the timestamp column.
        glucose_col: Name of the glucose values column.
        
    Raises:
        ValueError: If either required column is missing.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if glucose_col not in df.columns:
        raise ValueError(f"Missing glucose column: {glucose_col}")


def profile_cgm_series(df: pd.DataFrame, timestamp_col: str, glucose_col: str) -> CGMProfile:
    """Generate a comprehensive profile of a CGM time series.
    
    This function analyzes a CGM dataset to compute:
    - Temporal statistics (date range, cadence, gaps)
    - Data quality metrics (NaN count, non-numeric values, duplicates)
    - Per-day data point counts
    
    The function also normalizes timestamps and removes duplicate entries.
    
    Args:
        df: DataFrame containing CGM data.
        timestamp_col: Name of the timestamp column.
        glucose_col: Name of the glucose values column.
        
    Returns:
        CGMProfile object containing comprehensive dataset statistics.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    # TZ handling done by upstream cgm_parser - no need to repeat here
    duplicates_before = len(df)
    df = df.drop_duplicates(subset=[timestamp_col])
    duplicates_dropped = duplicates_before - len(df)
    df = df.sort_values(timestamp_col)

    # Coerce to numeric; keep NaNs (report them, do not drop)
    numeric = pd.to_numeric(df[glucose_col], errors="coerce")
    non_numeric_rows = int(np.sum(~df[glucose_col].astype(str).str.match(r"^[-+]?[0-9]*\.?[0-9]+$")))
    nan_count = int(numeric.isna().sum())

    # Time deltas
    ts = df[timestamp_col]
    deltas = ts.diff().dropna().dt.total_seconds()
    cadence_seconds_est = float(np.median(deltas)) if len(deltas) else None
    cadence_iqr_seconds = float(np.subtract(*np.percentile(deltas, [75, 25]))) if len(deltas) else None

    # Per-day counts based on timestamps
    per_day_counts = ts.dt.floor("D").value_counts().sort_index().rename_axis("date").to_frame("n_points")
    n_days = int(per_day_counts.shape[0])

    # Gap stats: gaps > 3x cadence
    num_gaps = 0
    max_gap_min = 0.0
    total_gap_min = 0.0
    if len(deltas) and cadence_seconds_est:
        threshold = 3.0 * cadence_seconds_est
        large = deltas[deltas > threshold]
        num_gaps = int(large.shape[0])
        max_gap_min = float(large.max() / 60.0) if large.shape[0] else 0.0
        total_gap_min = float(large.sum() / 60.0)

    gap_stats = {
        "num_gaps": num_gaps,
        "max_gap_min": max_gap_min,
        "total_gap_min": total_gap_min,
    }

    start_iso = ts.min().isoformat() if not ts.isna().all() else None
    end_iso = ts.max().isoformat() if not ts.isna().all() else None

    return CGMProfile(
        n_rows=len(df),
        n_days=n_days,
        start_iso=start_iso,
        end_iso=end_iso,
        cadence_seconds_est=cadence_seconds_est,
        cadence_iqr_seconds=cadence_iqr_seconds,
        per_day_counts=per_day_counts,
        gap_stats=gap_stats,
        nan_count=nan_count,
        non_numeric_rows=non_numeric_rows,
        duplicates_dropped=duplicates_dropped,
    )


