"""
CGM Metrics - Comprehensive glucose monitoring analysis

Computes 55 clinical metrics from CGM time-series data using the iglu_python library.
All metrics use standardized cgm_* naming.
Includes glycemic variability, time-in-range, clinical indices, and episode detection.

## When to Use
- Analyzing CGM time-series data
- Computing standardized clinical metrics
- Comparing glucose patterns across days
- Performance-sensitive workflows (use metrics_subset)

## When NOT to Use
- For parsing raw CGM files (use cgm_parser)
- For real-time monitoring (this is batch analysis)

## Requirements
- Requires CGM DataFrame with datetime index and glucose column
- Minimum 1 day of data
- Timezone-aware timestamps (preserved, not converted to UTC)

## Performance
- Runtime scales linearly with the number of days of data.
- Use metrics_subset to compute only the metrics you need.

## Example
    # All metrics directly accessible at top level
    result = cgm_metrics(cgm_aid="cgm_abc123", units="mg/dL")
    print(result.cgm_mean)          # Average glucose
    print(result.cgm_gmi)           # Glucose Management Indicator
    print(result.cgm_cv)            # Coefficient of variation
    print(result.days_valid)        # Data quality

    # All 55 metrics available: cgm_mage, cgm_in_range_70_180, cgm_above_180, etc.
    # Full dicts stored as artifacts for reference
    full_metrics = get_artifact(result.global_metrics_aid)
    daily_df = get_artifact(result.daily_metrics_aid)

    # Compute only essential metrics (faster)
    result = cgm_metrics(
        cgm_aid="cgm_abc123",
        metrics_subset=["cgm_mean", "cgm_cv", "cgm_gmi"]
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import iglu_python as iglu
from pha.artifact_store import get_artifact, put_artifact
from pha.tool_decorator import tool


METRICS_VERSION = "1.2.0"

# ============================================================================
# Pydantic Models for Typed I/O
# ============================================================================

class CGMMetricsMetadata(BaseModel):
    """Flat metadata output for cgm_metrics tool.

    Design Philosophy:
    - Top-level: the 11 most clinically significant metrics, returned as scalars
    - Artifacts: full 55-metric dict for comprehensive analysis

    Clinical Metrics Selection: the ADA/ATTD core metrics, plus MAGE and
    J-index for glycaemic variability:
    - Primary efficacy: Time in Range (70-180 mg/dL)
    - Glycemic control: Mean glucose, GMI (HbA1c proxy)
    - Safety: Hypoglycemia <70 and <54 mg/dL
    - Hyperglycemia: >180 and >250 mg/dL
    - Variability: CV% and SD

    Attributes:
        # Essential metadata
        days_valid: Days meeting coverage requirement
        days_total: Total days in dataset
        units: Glucose measurement units
        backend: Metrics computation backend

        # Clinically significant metrics (top-level)
        cgm_mean: Mean glucose value (primary control metric)
        cgm_gmi: Glucose Management Indicator (% eA1c proxy)
        cgm_cv: Coefficient of variation (% variability)
        cgm_sd: Standard deviation (absolute variability)
        cgm_in_range_70_180: Time in Range 70-180 mg/dL (% primary outcome)
        cgm_below_70: Time below 70 mg/dL (% hypoglycemia)
        cgm_below_54: Time below 54 mg/dL (% severe hypoglycemia)
        cgm_above_180: Time above 180 mg/dL (% hyperglycemia)
        cgm_above_250: Time above 250 mg/dL (% severe hyperglycemia)
        cgm_mage: Mean amplitude of glycemic excursions (variability)
        cgm_j_index: J-index (overall quality score)

        # Artifacts
        global_metrics_aid: Full metrics dict artifact (55 metrics)
        metric_units_aid: Metric name -> unit mapping artifact
        daily_metrics_aid: Per-day metrics DataFrame artifact
    """
    # Essential metadata
    days_valid: int = Field(description="Days meeting coverage requirement")
    days_total: int = Field(description="Total days in dataset")
    units: str = Field(description="Glucose measurement units (mg/dL or mmol/L)")
    backend: str = Field(description="Metrics computation backend", default="iglu-python")

    # Clinically significant metrics (ADA/ATTD core, plus MAGE and J-index)
    cgm_mean: float = Field(description="Mean glucose value")
    cgm_gmi: float = Field(description="Glucose Management Indicator (% eA1c)")
    cgm_cv: float = Field(description="Coefficient of variation (%)")
    cgm_sd: float = Field(description="Standard deviation")
    cgm_in_range_70_180: float = Field(description="Time in Range 70-180 mg/dL (%)")
    cgm_below_70: float = Field(description="Time below 70 mg/dL (%)")
    cgm_below_54: float = Field(description="Time below 54 mg/dL (% severe hypo)")
    cgm_above_180: float = Field(description="Time above 180 mg/dL (%)")
    cgm_above_250: float = Field(description="Time above 250 mg/dL (% severe hyper)")
    cgm_mage: float = Field(description="Mean amplitude glycemic excursions")
    cgm_j_index: float = Field(description="J-index quality score")

    # Artifacts
    global_metrics_aid: str = Field(description="Full metrics dict artifact (55 metrics)")
    metric_units_aid: str = Field(description="Metric units mapping artifact")
    daily_metrics_aid: str = Field(description="Per-day metrics DataFrame artifact")

# ============================================================================
# Metrics Catalog (CSV-based metadata)
# ============================================================================

IGLU_CATALOG_COLS = ['iglu_output_name', 'standardized_name', 'unit_category',
                     'category', 'short_name', 'description', 'iglu_function']

def _load_metrics_catalog() -> pd.DataFrame:
    """Load metrics catalog from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns: iglu_output_name, standardized_name,
            unit_category, category, short_name, description
    """
    catalog_path = Path(__file__).parent / "config" / "cgm_metrics_catalog.csv"
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Metrics catalog not found: {catalog_path}")
    
    df = pd.read_csv(catalog_path)
    
    # Validate required columns
    missing = set(IGLU_CATALOG_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")
    
    return df


# Load catalog at module level
METRICS_CATALOG = _load_metrics_catalog()

# Generate name mapping dict
CGM_METRIC_NAME_MAPPING: Dict[str, str] = dict(
    zip(METRICS_CATALOG['iglu_output_name'], METRICS_CATALOG['standardized_name'])
)

# Generate reverse mapping (used by the metrics_subset argument)
STANDARDIZED_TO_IGLU_MAPPING: Dict[str, List[str]] = (
    METRICS_CATALOG.groupby('standardized_name')['iglu_output_name']
    .apply(list)
    .to_dict()
)

# Generate category groupings
METRIC_CATEGORIES: Dict[str, List[str]] = (
    METRICS_CATALOG.groupby('category')['standardized_name']
    .apply(lambda x: sorted(x.unique()))
    .to_dict()
)

# Add essential category manually (most commonly used metrics)
METRIC_CATEGORIES['essential'] = [
    'cgm_mean', 'cgm_cv', 'cgm_gmi',
    'cgm_in_range_70_180', 'cgm_above_180', 'cgm_below_70',
]

# Generate unit category mappings
METRIC_UNIT_CATEGORIES: Dict[str, List[str]] = (
    METRICS_CATALOG.groupby('unit_category')['standardized_name']
    .apply(lambda x: sorted(x.unique()))
    .to_dict()
)

# Generate list of iglu function names from catalog (instead of hardcoding)
IGLU_METRIC_NAMES: List[str] = sorted(METRICS_CATALOG['iglu_function'].unique().tolist())

# Unit label formatters
UNIT_CATEGORY_LABELS = {
    'glucose': '{units}',
    'percentage': '%',
    'cv_percent': '% CV',
    'unitless': '',
    'a1c_percent': '%',
    'rate': '{units}/min',
    'auc': '{units}·h',
    'glucose_per_hour': '{units}/h',
}


def get_metric_unit(metric_name: str, glucose_units: str = "mg/dL") -> str:
    """Get unit label for a metric.

    Args:
        metric_name (str): Standardized metric name (e.g., 'cgm_mean')
        glucose_units (str): Base glucose units ('mg/dL' or 'mmol/L')

    Returns:
        str: Unit string (e.g., 'mg/dL', '%', '', 'mg/dL/min')
    """
    # Look up unit category from catalog
    matching = METRICS_CATALOG[METRICS_CATALOG['standardized_name'] == metric_name]
    
    if matching.empty:
        return ""  # Unknown metric
    
    unit_category = matching.iloc[0]['unit_category']
    label_template = UNIT_CATEGORY_LABELS.get(unit_category, '')
    
    return label_template.format(units=glucose_units)


def build_metric_units_dict(glucose_units: str = "mg/dL") -> Dict[str, str]:
    """Build complete metric_units dict for all known metrics.

    This is used for the metadata output.

    Args:
        glucose_units (str): Base glucose units ('mg/dL' or 'mmol/L')

    Returns:
        Dict[str, str]: Dictionary mapping metric names to unit strings
    """
    units_dict = {}
    
    for _, row in METRICS_CATALOG[['standardized_name']].drop_duplicates().iterrows():
        metric_name = row['standardized_name']
        units_dict[metric_name] = get_metric_unit(metric_name, glucose_units)
    
    # Special override for timestamp fields (have better description than empty string from unitless)
    units_dict["cgm_active_start_date"] = "ISO 8601 timestamp"
    units_dict["cgm_active_end_date"] = "ISO 8601 timestamp"
    
    return units_dict


def get_metric_description(metric_name: str) -> Optional[str]:
    """Get human-readable description for a metric.

    Args:
        metric_name (str): Standardized metric name (e.g., 'cgm_mean')

    Returns:
        Optional[str]: Description string or None if not found
    """
    matching = METRICS_CATALOG[METRICS_CATALOG['standardized_name'] == metric_name]
    
    if matching.empty:
        return None
    
    return matching.iloc[0]['description']


def get_metric_short_name(metric_name: str) -> Optional[str]:
    """Get short display name for a metric.

    Args:
        metric_name (str): Standardized metric name (e.g., 'cgm_mean')

    Returns:
        Optional[str]: Short name string or None if not found
    """
    matching = METRICS_CATALOG[METRICS_CATALOG['standardized_name'] == metric_name]
    
    if matching.empty:
        return None
    
    return matching.iloc[0]['short_name']


def get_metric_category(metric_name: str) -> Optional[str]:
    """Get category for a metric.

    Args:
        metric_name (str): Standardized metric name (e.g., 'cgm_mean')

    Returns:
        Optional[str]: Category string or None if not found
    """
    matching = METRICS_CATALOG[METRICS_CATALOG['standardized_name'] == metric_name]
    
    if matching.empty:
        return None
    
    return matching.iloc[0]['category']

# ============================================================================
# Helper Functions
# ============================================================================

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a DatetimeIndex, converting if necessary.

    Args:
        df (pd.DataFrame): Input DataFrame that may or may not have a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex.

    Raises:
        ValueError: If no datetime column is found.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    
    # Find first datetime column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df.set_index(col)
    
    raise ValueError("No datetime column found in DataFrame artifact")


def _identify_glucose_column(df: pd.DataFrame) -> str:
    """Identify the glucose value column in the DataFrame.

    Args:
        df (pd.DataFrame): CGM DataFrame.

    Returns:
        str: Name of the glucose column.
    """
    glucose_candidates = [c for c in df.columns if c.lower() in {"glucose", "cgm", "value"}]
    return glucose_candidates[0] if glucose_candidates else df.columns[0]


def _prepare_iglu_dataframe(
    df: pd.DataFrame,
    glucose_col: str,
    cgm_aid: str
) -> pd.DataFrame:
    """Prepare DataFrame in iglu format: ['id', 'time', 'gl'].

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex and glucose column.
        glucose_col (str): Name of glucose column.
        cgm_aid (str): Artifact ID to use as subject ID.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'time', 'gl'] ready for iglu_python.
    """
    # Convert to iglu format (without resampling)
    df_iglu = df[[glucose_col]].rename_axis("time").reset_index()
    df_iglu = df_iglu.rename(columns={glucose_col: "gl"})
    df_iglu["id"] = str(cgm_aid)
    return df_iglu[["id", "time", "gl"]]


def _process_single_day(
    day: pd.Timestamp,
    df_iglu: pd.DataFrame,
    expected_points_per_day: int,
    require_min_per_day: float,
    metrics_subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Process metrics for a single day.

    Args:
        day (pd.Timestamp): Date to process.
        df_iglu (pd.DataFrame): Full DataFrame in iglu format.
        expected_points_per_day (int): Expected number of points per day.
        require_min_per_day (float): Minimum coverage ratio to mark day as valid.
        metrics_subset (Optional[List[str]]): Optional list of metrics to compute. If None, computes all.

    Returns:
        Dict[str, Any]: Day record with metadata and metrics.
    """
    day_mask = df_iglu["time"].dt.floor("D") == day
    day_df = df_iglu.loc[day_mask].copy()
    
    n_points = int(day_df.shape[0])
    n_valid = int(day_df["gl"].notna().sum())
    coverage = float(n_valid / expected_points_per_day) if expected_points_per_day else 0.0
    
    metrics = _run_iglu_metrics(day_df, metrics_subset=metrics_subset)
    
    # Build day record with metadata and flattened metrics
    day_record = {
        "date": str(pd.Timestamp(day).date()),
        "n_points": n_points,
        "n_valid": n_valid,
        "expected_points": expected_points_per_day,
        "coverage_pct": coverage,
        "valid": bool(coverage >= require_min_per_day),
    }
    day_record.update(metrics)
    return day_record


def _convert_to_columnar(daily_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert list of daily dicts to columnar format (dict of arrays).

    Args:
        daily_list (List[Dict[str, Any]]): List of dicts where each dict represents one day's metrics.

    Returns:
        Dict[str, List[Any]]: Dictionary where each key maps to a list of values (one per day).

    Example:
        Input: [{"date": "2023-01-01", "cgm_cv": 30.5}, {"date": "2023-01-02", "cgm_cv": 32.1}]
        Output: {"date": ["2023-01-01", "2023-01-02"], "cgm_cv": [30.5, 32.1]}
    """
    if not daily_list:
        return {}
    
    # Collect all unique keys across all days
    all_keys = set()
    for day_dict in daily_list:
        all_keys.update(day_dict.keys())
    
    # Build columnar dict
    columnar: Dict[str, List[Any]] = {}
    for key in sorted(all_keys):
        columnar[key] = [day_dict.get(key) for day_dict in daily_list]
    
    return columnar


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types for JSON serialization.

    Args:
        value (Any): Value to convert (can be scalar, array, dict, list, etc.).

    Returns:
        Any: Native Python type equivalent (int, float, str, None, list, dict).
    """
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _normalize_metric_output(result: Any, metric_name: str) -> Any:
    """Normalize iglu_python metric outputs to consistent Python types.

    Args:
        result (Any): Raw output from an iglu_python metric function (DataFrame, Series, dict, etc.).
        metric_name (str): The name of the metric being normalized (used for special handling).

    Returns:
        Any: Normalized output (None for empty, dict for single-row DF, list for multi-row DF).
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        records: List[Dict[str, Any]] = []
        for rec in result.to_dict(orient="records"):
            processed = {k: _to_builtin(v) for k, v in rec.items() if k != "id"}
            if processed:
                records.append(processed)
        if not records:
            return None
        if len(records) == 1:
            return records[0]
        return records
    if isinstance(result, pd.Series):
        if result.empty:
            return None
        return {k: _to_builtin(v) for k, v in result.to_dict().items()}
    if isinstance(result, dict):
        return {k: _to_builtin(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_to_builtin(v) for v in result]
    return _to_builtin(result)


def _flatten_metric(metric_name: str, metric_value: Any) -> Dict[str, Any]:
    """Flatten a single metric result into a flat dictionary with standardized names.

    Args:
        metric_name (str): Name of the metric (e.g., 'cv_glu', 'quantile_glu').
        metric_value (Any): Normalized metric value (dict, list, or scalar).

    Returns:
        Dict[str, Any]: Flattened dictionary with standardized cgm_* keys.

    Note:
        Special handling:
        - quantile_glu: keys '0', '25', '50', '75', '100' become 'quantile_0', 'quantile_25', etc.
        - summary_glu: keys remain as-is ('Min.', '1st Qu.', etc.) without prefix
        - episode_calculation: remains as nested list (not flattened).
        - Metrics with single uppercase key (e.g., CV, MAGE): use key as-is without prefix
        - Other dict results: keys are prefixed with metric name if not already present
        - All keys are then mapped to standardized cgm_* names using CGM_METRIC_NAME_MAPPING
    """
    if metric_value is None:
        return {}
    
    # Special case: episode_calculation is a list of dicts, keep nested but use standardized name
    if metric_name == "episode_calculation" and isinstance(metric_value, list):
        standardized_name = CGM_METRIC_NAME_MAPPING.get(metric_name, metric_name)
        return {standardized_name: metric_value}
    
    # Special case: quantile_glu has numeric string keys that need prefixing
    if metric_name == "quantile_glu" and isinstance(metric_value, dict):
        raw_flattened = {f"quantile_{k}": v for k, v in metric_value.items()}
        return _apply_standardized_naming(raw_flattened)
    
    # Special case: summary_glu has descriptive keys like 'Min.', '1st Qu.'
    if metric_name == "summary_glu" and isinstance(metric_value, dict):
        return _apply_standardized_naming(metric_value)
    
    # Dict results: smart prefixing
    if isinstance(metric_value, dict):
        flattened = {}
        for k, v in metric_value.items():
            # If single key and it's all uppercase or mixed case, use as-is
            if len(metric_value) == 1 and (k.isupper() or any(c.isupper() for c in k)):
                flattened[k] = v
            # If the key already contains the metric name, don't duplicate
            elif k.lower().replace("_", "") in metric_name.lower().replace("_", ""):
                flattened[k] = v
            else:
                flattened[f"{metric_name}_{k}"] = v
        return _apply_standardized_naming(flattened)
    
    # Scalar or list: return as-is with metric name as key
    raw_result = {metric_name: metric_value}
    return _apply_standardized_naming(raw_result)


def _apply_standardized_naming(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Apply standardized cgm_* naming to raw metric keys.

    Maps raw iglu output names (e.g., 'Mean', 'CV', 'above_percent_above_140')
    to standardized names (e.g., 'cgm_mean', 'cgm_cv', 'cgm_above_140') using
    the CGM_METRIC_NAME_MAPPING from the catalog.

    Handles deduplication: if multiple raw keys map to the same standardized key,
    keeps only the first occurrence.

    Args:
        raw_metrics (Dict[str, Any]): Dictionary with raw metric names as keys.

    Returns:
        Dict[str, Any]: Dictionary with standardized cgm_* names as keys.
    """
    standardized = {}
    
    for raw_key, value in raw_metrics.items():
        # Map to standardized name (or keep raw key if not in mapping)
        std_key = CGM_METRIC_NAME_MAPPING.get(raw_key, raw_key)
        
        # Deduplication: only keep first occurrence of each standardized key
        if std_key not in standardized:
            standardized[std_key] = value
    
    return standardized


def _determine_iglu_functions_needed(metrics_subset: List[str]) -> List[str]:
    """Determine which iglu functions to run for requested metrics.

    Uses the catalog to look up which iglu functions produce the requested
    standardized metrics. This enables performance optimization by only running
    necessary functions.

    Args:
        metrics_subset (List[str]): List of standardized metric names (e.g., ['cgm_mean', 'cgm_cv'])

    Returns:
        List[str]: List of iglu function names to run (e.g., ['mean_glu', 'cv_glu'])

    Example:
        >>> _determine_iglu_functions_needed(['cgm_mean', 'cgm_cv'])
        ['cv_glu', 'mean_glu', 'summary_glu']  # summary_glu also produces Mean
    """
    if metrics_subset is None:
        return IGLU_METRIC_NAMES
    
    # If empty list, return empty (no functions needed)
    if not metrics_subset:
        return []
    
    # Filter catalog to only rows whose standardized_name is in metrics_subset
    relevant_rows = METRICS_CATALOG[METRICS_CATALOG['standardized_name'].isin(metrics_subset)]
    
    # Get unique iglu functions needed
    functions_needed = set(relevant_rows['iglu_function'].unique())
    
    # Filter to only functions that exist in IGLU_METRIC_NAMES
    valid_functions = [f for f in functions_needed if f in IGLU_METRIC_NAMES]
    
    return sorted(valid_functions)


def _run_iglu_metrics(
    data: pd.DataFrame,
    metrics_subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run iglu_python metrics on the provided CGM data and flatten results.

    Optionally filters to a subset of metrics for improved performance.

    Args:
        data (pd.DataFrame): CGM data with columns ['id', 'time', 'gl'].
        metrics_subset (Optional[List[str]]): Optional list of standardized metric names to compute. If None, computes all.

    Returns:
        Dict[str, Any]: Flattened dictionary of computed metrics with standardized cgm_* names.
    """
    metrics: Dict[str, Any] = {}
    if data.empty:
        return metrics
    processed = iglu.process_data(data, id="id", timestamp="time", glu="gl")
    if processed.empty:
        return metrics
    
    # Determine which functions to run
    if metrics_subset is not None:
        metrics_to_run = _determine_iglu_functions_needed(metrics_subset)
    else:
        metrics_to_run = IGLU_METRIC_NAMES
    
    # Run the selected functions
    for name in metrics_to_run:
        fn = getattr(iglu, name)
        try:
            result = _normalize_metric_output(fn(processed), name)
            flattened = _flatten_metric(name, result)
            metrics.update(flattened)
        except Exception as exc:  # pragma: no cover - safeguard against upstream changes
            metrics[f"{name}_error"] = str(exc)
    
    # Filter output to only requested metrics (if subset specified)
    if metrics_subset:
        # Keep only metrics that were requested
        # Also keep special fields like active_percent_* and episode calculation
        metrics = {k: v for k, v in metrics.items() if k in metrics_subset}
    
    return metrics


@tool(version=METRICS_VERSION, categories=["CGM"], description="Compute comprehensive CGM metrics using iglu-python", depends_on=["cgm_parser"], mandatory=True)
def cgm_metrics(
    cgm_aid: Annotated[str, Field(description="Artifact ID for CGM DataFrame (cgm_* prefix)")],
    *,
    units: Annotated[
        Literal["mg/dL", "mmol/L"], 
        Field(description="Glucose measurement units")
    ] = "mg/dL",
    require_min_per_day: Annotated[
        float, 
        Field(description="Minimum coverage ratio (0-1) required to mark a day as valid", ge=0.0, le=1.0)
    ] = 0.7,
    metrics_subset: Annotated[
        Optional[List[str]],
        Field(description="Optional list of specific metrics to compute (e.g., ['cgm_mean', 'cgm_cv', 'cgm_gmi']). "
                          "If None, computes all metrics. Use standardized cgm_* names for improved performance.")
    ] = None,
) -> CGMMetricsMetadata:
    """Compute comprehensive CGM metrics using iglu_python library.

    This tool processes continuous glucose monitoring (CGM) time series data and computes
    a comprehensive suite of metrics for both daily and global (multi-day) analysis.
    All metrics are computed using the iglu_python library implementation.

    The tool:
    1. Loads CGM time series from artifact store by ID
    2. Identifies timestamp and glucose columns
    3. Computes coverage metrics for each day
    4. Runs all iglu_python metrics on each day and globally
    5. Returns flattened metric dictionaries for easy consumption

    Metrics are flattened and use standardized cgm_* naming. For example:
    - 'mean_glu' → {'mean': 120.5} becomes {'cgm_mean': 120.5}
    - 'quantile_glu' → {'0': 80, '50': 120} becomes {'cgm_min': 80, 'cgm_median': 120}
    - 'episode_calculation' remains nested as a list and is named 'cgm_episodes'

    **Daily Metrics Output:**
    Daily metrics are saved as a DataFrame artifact (with cgm_daily_* prefix).
    The DataFrame has one row per day with columns for date, coverage metadata,
    and all computed cgm_* metrics. Access via the returned `daily_metrics_aid`.

    **IMPORTANT - Metric Units:**
    The `metric_units` field in metadata provides the units/format for each metric.
    Pay special attention to percentage metrics:
    - Metrics like `above_percent_above_140`, `below_percent_below_70`, etc. are returned
      as **percentages** on a 0-100 scale. Use the values as-is.
      Example: 0.5608 means 0.56%, NOT 56.08% (do NOT multiply by 100)
    - Values less than 1.0 represent small percentages (< 1%)
    Always check `metric_units` to avoid misinterpreting metric values.

    Args:
        cgm_aid (str): Artifact ID for CGM DataFrame (cgm_* prefix)
        units (Literal["mg/dL", "mmol/L"]): Glucose measurement units (default: "mg/dL")
        require_min_per_day (float): Minimum coverage ratio (0-1) to mark day as valid (default: 0.7)
        metrics_subset (Optional[List[str]]): Optional list of specific metrics to compute for performance optimization

    Returns:
        CGMMetricsMetadata: Comprehensive metrics for both daily and global analysis

    Raises:
        ValueError: If no datetime column is found in the artifact DataFrame

    Usage:
        # Direct call - 11 key metrics at top level
        metadata = cgm_metrics(cgm_aid="cgm_abc123", units="mg/dL")
        print(metadata.cgm_mean)          # Direct access to key metric
        print(metadata.cgm_in_range_70_180)  # TIR - primary outcome
        print(metadata.days_valid)        # Essential metadata

        # When to use top-level metrics vs artifacts (Python Zen: "One obvious way"):
        #
        # Use top-level for:
        #   ✓ Common clinical questions (mean, TIR, GMI, hypo risk)
        #   ✓ Fast access (no artifact retrieval needed)
        #   ✓ 90% of use cases
        #
        # Use artifacts for:
        #   ✓ Niche/research metrics (CONGA, MODD, SDBDM)
        #   ✓ Daily breakdowns (per-day analysis)
        #   ✓ Complete metric reference
        #
        # Access artifacts when needed:
        full_metrics = get_artifact(metadata.global_metrics_aid)  # All 55+ metrics
        daily_df = get_artifact(metadata.daily_metrics_aid)       # Per-day breakdown
        units_map = get_artifact(metadata.metric_units_aid)       # Units reference

        # Framework call (returns envelope with provenance)
        envelope = cgm_metrics.invoke({"cgm_aid": "cgm_abc123", "units": "mg/dL"})
        print(envelope.metadata.cgm_mean)
        print(envelope.metadata.cgm_gmi)
        print(envelope.provenance.tool_version)

    See Also:
        iglu_python: https://github.com/staskh/iglu_python
    """
    # Step 1: Load and preprocess CGM data
    df = get_artifact(cgm_aid, expected_prefix="cgm")
    df = _ensure_datetime_index(df)
    df = df.sort_index()
    # Timezone handled upstream by cgm_parser

    # Step 2: Identify glucose column
    glucose_col = _identify_glucose_column(df)
    
    # Step 3: Prepare data in iglu format (no resampling)
    df_iglu = _prepare_iglu_dataframe(df, glucose_col, cgm_aid)
    
    # Step 4: Compute expected points per day from actual data
    # Use median number of points per day as the expected value
    points_per_day = df_iglu.groupby(df_iglu["time"].dt.floor("D")).size()
    expected_points_per_day = int(points_per_day.median()) if len(points_per_day) > 0 else 0
    
    # Step 5: Identify days to analyze
    all_days = pd.Index(sorted(df_iglu["time"].dt.floor("D").unique()))
    days_total = int(all_days.shape[0])
    selected_days = all_days  # Process all days (no arbitrary limit)

    # Step 6: Process metrics for each day
    daily_data: List[Dict[str, Any]] = [
        _process_single_day(day, df_iglu, expected_points_per_day, require_min_per_day, metrics_subset)
        for day in selected_days
    ]
    days_valid = int(sum(1 for d in daily_data if d.get("valid")))
    
    # Step 7: Compute global metrics across VALID days only
    valid_day_indices = [i for i, d in enumerate(daily_data) if d.get("valid")]
    if valid_day_indices:
        valid_days_timestamps = [selected_days[i] for i in valid_day_indices]
        valid_mask = df_iglu["time"].dt.floor("D").isin(valid_days_timestamps)
        global_metrics = _run_iglu_metrics(df_iglu.loc[valid_mask].copy(), metrics_subset=metrics_subset)
    else:
        global_metrics = {}

    # Step 8: Convert daily metrics to DataFrame and save as artifact
    daily_metrics_columnar: Dict[str, List[Any]] = _convert_to_columnar(daily_data)
    daily_metrics_df = pd.DataFrame(daily_metrics_columnar)
    daily_metrics_aid = put_artifact(daily_metrics_df, prefix="df")

    # Step 9: Store full dicts as artifacts
    global_metrics_aid = put_artifact(global_metrics, prefix="metrics")
    metric_units_dict = build_metric_units_dict(units)
    metric_units_aid = put_artifact(metric_units_dict, prefix="metrics")

    # Step 10: Extract only clinically significant metrics for top-level
    # Default to 0.0 for missing metrics (can happen if no valid days)
    key_metrics = {
        "cgm_mean": global_metrics.get("cgm_mean", 0.0),
        "cgm_gmi": global_metrics.get("cgm_gmi", 0.0),
        "cgm_cv": global_metrics.get("cgm_cv", 0.0),
        "cgm_sd": global_metrics.get("cgm_sd", 0.0),
        "cgm_in_range_70_180": global_metrics.get("cgm_in_range_70_180", 0.0),
        "cgm_below_70": global_metrics.get("cgm_below_70", 0.0),
        "cgm_below_54": global_metrics.get("cgm_below_54", 0.0),
        "cgm_above_180": global_metrics.get("cgm_above_180", 0.0),
        "cgm_above_250": global_metrics.get("cgm_above_250", 0.0),
        "cgm_mage": global_metrics.get("cgm_mage", 0.0),
        "cgm_j_index": global_metrics.get("cgm_j_index", 0.0),
    }

    # Step 11: Build and return flat metadata with key metrics at top level
    return CGMMetricsMetadata(
        # Essential metadata
        days_valid=days_valid,
        days_total=days_total,
        units=units,
        backend="iglu-python",
        # Artifacts
        global_metrics_aid=global_metrics_aid,
        metric_units_aid=metric_units_aid,
        daily_metrics_aid=daily_metrics_aid,
        # Key clinical metrics
        **key_metrics,
    )


