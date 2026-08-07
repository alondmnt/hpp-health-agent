"""
CGM Parser - Load and validate CGM CSV data

Loads CGM data from CSV files or strings, validates format, profiles data quality,
converts units to mg/dL, and stores in artifact store for downstream analysis.

## When to Use
- Loading raw CGM data from CSV files
- Converting mmol/L to mg/dL units
- Profiling data quality (gaps, duplicates, cadence)
- Preparing data for cgm_metrics analysis

## When NOT to Use
- For computing metrics (use cgm_metrics after parsing)
- For real-time data ingestion (this is batch processing)

## Requirements
- CSV file or string with timestamp and glucose columns
- Valid timezone string for localization
- Minimum 1 day of data recommended

## Performance
- Runtime is dominated by pandas CSV parsing; a 14-day trace is sub-second.

## Example
    # Defaults match the column names used by the bundled synthetic data
    result = cgm_parser(csv_path_or_buffer="data/synthetic_001/cgm.csv")
    print(result.cgm_aid)   # Pass this to cgm_metrics
    print(result.n_days)    # Number of days loaded

    # Or name the columns explicitly
    result = cgm_parser(
        csv_path_or_buffer="data/libre_export.csv",
        timestamp_col="Device Timestamp",
        glucose_col="Historic Glucose mmol/L",
        tz="America/New_York",
    )
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
import io
from pydantic import BaseModel, Field

from pha.artifact_store import put_artifact
from pha.tool_decorator import tool
from pha.utils import DEFAULT_TIMEZONE, localize_timestamps
from pha.validators import CGMProfile, profile_cgm_series, validate_cgm_dataframe


PARSER_VERSION = "0.0.1"


# ============================================================================
# Pydantic Models for Typed I/O
# ============================================================================

class CGMParserMetadata(BaseModel):
    """Typed metadata output for cgm_parser tool.

    Follows flat metadata pattern (Python Zen: "Flat is better than nested").

    Attributes:
        cgm_aid: Artifact ID for the parsed CGM DataFrame (cgm_* prefix)
        source_path: Optional path to source file (if loaded from file)
        original_units: Detected or specified input units
        conversion_factor: Factor applied for unit conversion (18.0 for mmol/L → mg/dL)
        n_rows: Number of data rows loaded
        duplicates_dropped: Number of duplicate timestamps removed
        n_days: Number of calendar days spanned
        start_iso: ISO timestamp of first data point (in specified timezone)
        end_iso: ISO timestamp of last data point (in specified timezone)
        sampling_interval_seconds: Estimated time between measurements in seconds (median)
        per_day_counts: Data point counts per day (context, not facts)
        gap_count: Number of gaps in time series (flattened from gap_stats)
        gap_max_minutes: Maximum gap duration in minutes (flattened from gap_stats)
        gap_total_minutes: Total gap time in minutes (flattened from gap_stats)
        nan_count: Number of missing glucose values
    """
    cgm_aid: str = Field(description="Artifact ID for the parsed CGM DataFrame (cgm_* prefix)")
    source_path: Optional[str] = Field(description="Path to source file (if applicable)", default=None)
    original_units: str = Field(description="Detected or specified input units")
    conversion_factor: float = Field(description="Factor applied for unit conversion")
    n_rows: int = Field(description="Number of data rows loaded")
    duplicates_dropped: int = Field(description="Number of duplicate timestamps removed")
    n_days: int = Field(description="Number of calendar days spanned")
    start_iso: str = Field(description="ISO timestamp of first data point (in specified timezone)")
    end_iso: str = Field(description="ISO timestamp of last data point (in specified timezone)")
    sampling_interval_seconds: Optional[float] = Field(description="Estimated time between measurements in seconds (median)", default=None)
    per_day_counts: List[Dict[str, Any]] = Field(description="Data point counts per day")
    gap_count: int = Field(description="Number of gaps in time series")
    gap_max_minutes: float = Field(description="Maximum gap duration in minutes")
    gap_total_minutes: float = Field(description="Total gap time in minutes")
    nan_count: int = Field(description="Number of missing glucose values")


# ============================================================================
# Helper Functions
# ============================================================================

def _detect_units(glucose_col: str, explicit: Optional[str]) -> str:
    """
    Detect glucose measurement units from column name or explicit specification.

    Parameters
    ----------
    glucose_col : str
        Name of the glucose column.
    explicit : Optional[str]
        Explicitly specified units (takes precedence if provided).

    Returns
    -------
    str
        Detected or specified units: 'mmol/L' if 'mmol' in column name, else 'mg/dL'.
    """
    if explicit:
        return explicit
    name = glucose_col.lower()
    if "mmol" in name:
        return "mmol/L"
    return "mg/dL"


def _maybe_convert_to_mgdl(series: pd.Series, units: str) -> tuple[pd.Series, float, str]:
    """
    Convert glucose values to mg/dL if needed.

    Parameters
    ----------
    series : pd.Series
        Glucose values to convert.
    units : str
        Current units of the glucose values.

    Returns
    -------
    tuple[pd.Series, float, str]
        - Converted series (or original if already in mg/dL)
        - Conversion factor applied (18.0 for mmol/L → mg/dL, 1.0 otherwise)
        - Output units ('mg/dL' or original if unrecognized)

    Notes
    -----
    Standard conversion: 1 mmol/L = 18 mg/dL
    """
    if units.lower() in {"mg/dl", "mgdl"}:
        return series, 1.0, "mg/dL"
    if units.lower() in {"mmol/l", "mmol"}:
        return series * 18.0, 18.0, "mg/dL"
    return series, 1.0, units


@tool(version=PARSER_VERSION, categories=["CGM", "DataProcessing"], mandatory=True, description="Parse and validate CGM CSV data, store in artifact store")
def cgm_parser(
    csv_path_or_buffer: Annotated[Optional[str], Field(description="File path to CSV (mutually exclusive with csv_string)")] = None,
    *,
    csv_string: Annotated[Optional[str], Field(description="CSV content as string (mutually exclusive with csv_path_or_buffer)")] = None,
    timestamp_col: Annotated[str, Field(description="Name of timestamp column")] = "collection_timestamp",
    glucose_col: Annotated[str, Field(description="Name of glucose value column")] = "glucose",
    tz: Annotated[str, Field(description="Timezone for timestamp localization")] = DEFAULT_TIMEZONE,
    units: Annotated[Optional[Literal["mg/dL", "mmol/L"]], Field(description="Explicit glucose units (if None, auto-detected)")] = None,
) -> CGMParserMetadata:
    """
    Parse and validate CGM data from CSV, store in artifact store.

    This tool loads continuous glucose monitoring data from a CSV file or string buffer,
    validates the format, profiles data quality, converts units to mg/dL, and stores
    the processed DataFrame in the artifact store for downstream analysis.

    Data Processing Pipeline:
    1. Load CSV from file or string buffer
    2. Validate required columns exist
    3. Parse timestamps and normalize to specified timezone (for cross-sensor alignment)
    4. Detect or use explicit glucose units
    5. Convert mmol/L to mg/dL if needed (×18.0)
    6. Profile data quality (gaps, duplicates, cadence)
    7. Sort by timestamp
    8. Store in artifact store
    9. Return metadata and artifact ID

    The artifact ID can be used with cgm_metrics or other downstream tools.

    Args:
        csv_path_or_buffer: File path to CSV (mutually exclusive with csv_string)
        csv_string: CSV content as string (mutually exclusive with csv_path_or_buffer)
        timestamp_col: Name of timestamp column (default: "collection_timestamp")
        glucose_col: Name of glucose value column (default: "glucose")
        tz: Timezone for timestamp normalization (default: "UTC")
        units: Explicit glucose units - 'mg/dL' or 'mmol/L'. If None, auto-detected from column name.

    Returns:
        CGMParserMetadata with profiling statistics and artifact ID

    Raises:
        ValueError: If required columns are missing or data validation fails
        FileNotFoundError: If csv_path_or_buffer points to non-existent file
        
    Examples:
        Load from file with explicit units:
        >>> result = cgm_parser(
        ...     csv_path_or_buffer="data/libre_export.csv",
        ...     timestamp_col="Device Timestamp",
        ...     glucose_col="Historic Glucose mmol/L",
        ...     tz="America/New_York",
        ...     units="mmol/L"
        ... )
        >>> ts_aid = result.cgm_aid

        Load from a string buffer using the default column names:
        >>> csv_data = "collection_timestamp,glucose\\n2024-01-01 00:00,120\\n"
        >>> result = cgm_parser(csv_string=csv_data, tz="UTC")
    """
    # Load CSV from file or string buffer
    if isinstance(csv_string, str):
        df = pd.read_csv(io.StringIO(csv_string))
    else:
        df = pd.read_csv(csv_path_or_buffer)

    validate_cgm_dataframe(df, timestamp_col, glucose_col)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[timestamp_col] = localize_timestamps(df[timestamp_col], tz)

    # Profile data quality
    profile: CGMProfile = profile_cgm_series(df, timestamp_col, glucose_col)

    # Detect and convert units
    original_units = _detect_units(glucose_col, units)
    converted_series, conversion_factor, _ = _maybe_convert_to_mgdl(df[glucose_col], original_units)
    df[glucose_col] = converted_series

    # Sort and store with CGM prefix
    df = df.sort_values(timestamp_col)
    ts_aid = put_artifact(df, prefix="cgm")

    # Format per-day counts
    per_day_counts = [
        {"date": str(idx.date()), "n_points": int(row["n_points"])}
        for idx, row in profile.per_day_counts.iterrows()
    ]

    # Determine source path (if applicable)
    source_path = (csv_path_or_buffer 
                   if (isinstance(csv_path_or_buffer, str) and not isinstance(csv_string, str)) 
                   else None)

    # Build and return typed metadata with flattened gap stats
    return CGMParserMetadata(
        cgm_aid=ts_aid,
        source_path=source_path,
        original_units=original_units,
        conversion_factor=conversion_factor,
        n_rows=profile.n_rows,
        duplicates_dropped=profile.duplicates_dropped,
        n_days=profile.n_days,
        start_iso=profile.start_iso,
        end_iso=profile.end_iso,
        sampling_interval_seconds=profile.cadence_seconds_est,
        per_day_counts=per_day_counts,
        # Flatten gap_stats (Python Zen: "Flat is better than nested")
        gap_count=profile.gap_stats.get("num_gaps", 0),
        gap_max_minutes=profile.gap_stats.get("max_gap_min", 0.0),
        gap_total_minutes=profile.gap_stats.get("total_gap_min", 0.0),
        nan_count=profile.nan_count,
    )


