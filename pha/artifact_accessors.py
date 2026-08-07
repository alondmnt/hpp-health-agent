"""Bounded artifact accessors for agent reasoning.

Provides token-limited views of artifacts without loading full data into context.
All accessors enforce strict size limits to prevent context bloat.

Design Philosophy (Python Zen):
    - Simple functions, no classes
    - Explicit size limits (no surprises)
    - Clear errors with helpful hints
    - ~100 lines total

Usage:
    >>> from pha.artifact_accessors import get_metric, df_head
    >>>
    >>> # Get specific metric
    >>> mean = get_metric("metrics_abc123", "Mean_mgdl")
    >>>
    >>> # Preview DataFrame
    >>> preview = df_head("cgm_xyz", n=10)
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .artifact_store import get_artifact


def get_metric(artifact_id: str, metric_name: str) -> float | int | str | bool | str:
    """Get single metric value from metrics artifact.

    Args:
        artifact_id: Metrics artifact ID (e.g., "metrics_abc123").
        metric_name: Metric key (e.g., "Mean_mgdl", "CV_percent").

    Returns:
        Metric value (scalar: int, float, str, or bool), or error string if failed.

    Examples:
        >>> mean_glucose = get_metric("metrics_xyz", "cgm_mean")
        >>> # Returns: 117.2
        >>>
        >>> result = get_metric("meal_report_xyz", "cgm_mean")
        >>> # Returns: "Error: get_metric() expects dict artifact, got str..."

    Note:
        Returns error strings instead of raising exceptions so ReactAnalyzer
        can read the error and try a different approach.
    """
    try:
        # Get artifact (let artifact_store handle basic validation)
        metrics = get_artifact(artifact_id)

        # Validate metrics is a dict
        if not isinstance(metrics, dict):
            return (
                f"Error: get_metric() expects dict artifact, got {type(metrics).__name__}. "
                f"Artifact '{artifact_id}' is not a metrics dict. "
                f"Try text_read() for lists or df_head() for DataFrames."
            )

        # Check metric exists
        if metric_name not in metrics:
            available = list(metrics.keys())[:10]
            suffix = "..." if len(metrics) > 10 else ""
            return (
                f"Error: Metric '{metric_name}' not found in {artifact_id}. "
                f"Available metrics: {available}{suffix}. Use list_metrics('{artifact_id}') to see all."
            )

        value = metrics[metric_name]

        # Ensure scalar (not nested dict/list)
        if not isinstance(value, (int, float, str, bool)):
            return (
                f"Error: Metric '{metric_name}' is not a scalar (got {type(value).__name__}). "
                f"Use text_read('{artifact_id}') to see the full nested structure."
            )

        return value

    except Exception as e:
        return f"Error in get_metric('{artifact_id}', '{metric_name}'): {str(e)}"


def list_metrics(artifact_id: str, limit: int = 50) -> list[str] | str:
    """List available metric names in artifact.

    Args:
        artifact_id: Metrics artifact ID.
        limit: Max metric names to return (default: 50).

    Returns:
        List of metric names (limited to `limit`), or error string if failed.

    Examples:
        >>> keys = list_metrics("metrics_xyz")
        >>> # Returns: ["cgm_mean", "cgm_sd", "cgm_cv", ...]
        >>>
        >>> result = list_metrics("cgm_dataframe_xyz")
        >>> # Returns: "Error: list_metrics() expects dict artifact, got DataFrame..."

    Note:
        Returns error strings instead of raising exceptions so ReactAnalyzer
        can read the error and try a different approach.
    """
    try:
        # Get artifact
        metrics = get_artifact(artifact_id)

        if not isinstance(metrics, dict):
            return (
                f"Error: list_metrics() expects dict artifact, got {type(metrics).__name__}. "
                f"Artifact '{artifact_id}' is not a metrics dict. "
                f"Try text_read() for lists or df_schema() for DataFrames."
            )

        # Return limited list
        all_keys = list(metrics.keys())
        return all_keys[:limit]

    except Exception as e:
        return f"Error in list_metrics('{artifact_id}'): {str(e)}"


def df_head(artifact_id: str, n: int = 50) -> dict[str, list] | str:
    """Get first N rows of DataFrame artifact.

    Args:
        artifact_id: DataFrame artifact ID (e.g., "cgm_abc", "df_xyz").
        n: Number of rows to return (max: 100, enforced).

    Returns:
        Dict with column names as keys, lists of values, or error string if failed.

    Examples:
        >>> preview = df_head("cgm_abc123", n=10)
        >>> # Returns: {"collection_timestamp": [...], "glucose": [...]}
        >>>
        >>> result = df_head("meal_report_xyz", n=5)
        >>> # Returns: "Error: Expected DataFrame, got str. Try text_read() instead."

    Note:
        Returns error strings instead of raising exceptions so ReactAnalyzer
        can read the error and try a different approach.
    """
    try:
        # Enforce strict limit to prevent context bloat
        if n > 100:
            return (
                f"Error: n must be ≤100 (bounded accessor). Got: {n}. "
                f"Use get_artifact() directly if you need full data."
            )

        # Get artifact
        df = get_artifact(artifact_id)

        # Validate DataFrame type
        if not isinstance(df, pd.DataFrame):
            return (
                f"Error: df_head() expects DataFrame, got {type(df).__name__}. "
                f"Artifact '{artifact_id}' is not a DataFrame. Try get_metric() for dicts "
                f"or text_read() for text artifacts."
            )

        # Return as dict (JSON-serializable)
        return df.head(n).to_dict(orient="list")

    except Exception as e:
        return f"Error in df_head('{artifact_id}', n={n}): {str(e)}"


def df_schema(artifact_id: str) -> dict[str, Any] | str:
    """Get DataFrame structure without data.

    Args:
        artifact_id: DataFrame artifact ID.

    Returns:
        Dict with columns, dtypes, shape, memory_mb, or error string if failed.

    Examples:
        >>> schema = df_schema("cgm_abc")
        >>> # Returns: {
        >>> #   "columns": ["collection_timestamp", "glucose"],
        >>> #   "dtypes": {"collection_timestamp": "datetime64[ns, UTC]", "glucose": "float64"},
        >>> #   "shape": [1171, 3],
        >>> #   "memory_mb": 0.05
        >>> # }

    Note:
        Returns error strings instead of raising exceptions so ReactAnalyzer
        can read the error and try a different approach.
    """
    try:
        # Get artifact
        df = get_artifact(artifact_id)

        # Validate DataFrame type
        if not isinstance(df, pd.DataFrame):
            return (
                f"Error: df_schema() expects DataFrame, got {type(df).__name__}. "
                f"Artifact '{artifact_id}' is not a DataFrame."
            )

        # Build schema info
        memory_bytes = df.memory_usage(deep=True).sum()

        return {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": list(df.shape),
            "memory_mb": round(memory_bytes / 1024 / 1024, 3),
        }

    except Exception as e:
        return f"Error in df_schema('{artifact_id}'): {str(e)}"


def text_read(artifact_id: str, max_chars: int = 10000) -> str:
    """Read text, dict, or list artifact with size limit.

    Args:
        artifact_id: Artifact ID (e.g., "meal_report_xyz", "metrics_abc").
        max_chars: Maximum characters to return (default: 10000, max: 50000).

    Returns:
        Text content (truncated if exceeds max_chars), or error string if failed.
        For dict/list artifacts, returns JSON-formatted string.

    Examples:
        >>> report = text_read("meal_report_abc123")
        >>> # Returns: "## Dietary Analysis\\n\\nYour meal patterns..."
        >>>
        >>> metrics = text_read("metrics_xyz")
        >>> # Returns: '{"cgm_mean": 105.3, "cgm_cv": 0.15, ...}'
        >>>
        >>> qc = text_read("df_qc_findings_abc")
        >>> # Returns: '[{"rule": "atwater_consistency", ...}, ...]'

    Note:
        This accessor handles text, dict, and list artifacts. Dict/list are
        JSON-serialized for readable output. Use get_metric() for accessing
        individual scalar values from dicts.

        Returns error strings instead of raising exceptions so ReactAnalyzer
        can read the error and try a different approach.
    """
    try:
        # Enforce strict limit to prevent context bloat
        if max_chars > 50000:
            return (
                f"Error: max_chars must be ≤50000 (bounded accessor). Got: {max_chars}. "
                f"Use get_artifact() directly if you need full text."
            )

        # Get artifact
        obj = get_artifact(artifact_id)

        # Convert to text based on type
        if isinstance(obj, str):
            text = obj
        elif isinstance(obj, (dict, list)):
            text = json.dumps(obj, indent=2, default=str)
        else:
            return (
                f"Error: text_read() expects str/dict/list, got {type(obj).__name__}. "
                f"Artifact '{artifact_id}' is not readable as text. "
                f"Use df_head() for DataFrames."
            )

        # Truncate if necessary
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[Truncated: {len(text) - max_chars} more characters]"

        return text

    except Exception as e:
        return f"Error in text_read('{artifact_id}'): {str(e)}"


# Public API
__all__ = [
    "get_metric",
    "list_metrics",
    "df_head",
    "df_schema",
    "text_read",
]
