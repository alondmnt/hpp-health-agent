"""File-based artifact store with type validation and session management.

This module provides a file-based store for artifacts (DataFrames, dicts, figures)
that can be referenced between tool calls via file paths. Artifacts are persisted
to disk with optimal formats (Parquet, JSON) and validated against type constraints.

ARTIFACT NAMING CONVENTIONS:
- All artifact ID variables/args should use `_aid` suffix
- Artifact IDs are file paths (e.g., "./artifacts/cgm_abc123.parquet")
- Prefix indicates artifact type:
  * "cgm": pandas.DataFrame (CGM time series data)
  * "df": pandas.DataFrame (generic)
  * "metrics": dict (computed metrics)
  * "plot": BytesIO or matplotlib.Figure (visualizations)
  * "obj": Any (untyped)

USAGE:
    # Store artifact - returns file path
    cgm_aid = put_artifact(dataframe, prefix="cgm")
    # Returns: "./artifacts/cgm_abc123.parquet"

    # Retrieve with prefix validation
    df = get_artifact(cgm_aid, expected_prefix="cgm")

    # Session management (optional)
    from pha.artifact_store import set_artifact_session
    set_artifact_session("./artifacts/experiment_001/")

PERSISTENCE:
    Artifacts are stored as files and survive restarts. File format is automatically
    selected based on type: Parquet for DataFrames, JSON for dicts, PNG for figures.
"""
from __future__ import annotations

import io
import json
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, Type, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


# ============================================================================
# Custom Exceptions
# ============================================================================

class ArtifactNotFoundError(KeyError):
    """Raised when an artifact ID is not found in the store.
    
    This is a subclass of KeyError for backward compatibility but provides
    a more specific exception type for artifact-related errors.
    """
    pass


class ArtifactTypeMismatchError(TypeError):
    """Raised when artifact type doesn't match expected type/prefix.
    
    This is a subclass of TypeError for backward compatibility but provides
    a more specific exception type for artifact type validation errors.
    """
    pass

try:
    from matplotlib.figure import Figure  # optional
except Exception:  # pragma: no cover
    class Figure:  # minimal stub if matplotlib isn't installed
        pass


# File-based storage: no in-memory dict needed, artifacts are files on disk.


@dataclass
class ArtifactInfo:
    """Metadata about a stored artifact."""
    aid: str
    prefix: str
    type_name: str
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None


def _normalize_types(t: Type | Tuple[Type, ...]) -> Tuple[Type, ...]:
    """Normalize type specification to tuple format."""
    return t if isinstance(t, tuple) else (t,)


# Prefix -> expected Python type(s) (always normalized to tuple[type, ...])
TYPE_REGISTRY: Dict[str, Tuple[Type, ...]] = {
    "cgm": _normalize_types(pd.DataFrame),
    "df": _normalize_types(pd.DataFrame),
    "metrics": _normalize_types(dict),
    "plot": _normalize_types((io.BytesIO, Figure)),  # BytesIO or Matplotlib Figure
    # note: leave "obj" unregistered to accept anything by default
}


def register_type(prefix: str, types: Type | Tuple[Type, ...]) -> None:
    """Register or override expected Python type(s) for a prefix.

    Args:
        prefix: Type prefix (e.g., "model", "embeddings")
        types: Expected Python type(s) for this prefix

    Examples:
        >>> from sklearn.base import BaseEstimator
        >>> register_type("model", BaseEstimator)
        >>> register_type("plot", (io.BytesIO, Figure))
    """
    TYPE_REGISTRY[prefix] = _normalize_types(types)


# ============================================================================
# Session Management
# ============================================================================

def set_artifact_session(session_dir: Optional[str] = None) -> str:
    """Set the artifact directory for the current session.

    Args:
        session_dir: Directory path for artifacts. If None, generates a timestamped session.

    Returns:
        The resolved absolute path that was set

    Examples:
        >>> # Auto-generate session
        >>> set_artifact_session()
        '/path/to/artifacts/session_20251124_143022'

        >>> # Named session
        >>> set_artifact_session("./artifacts/experiment_001")
        '/path/to/artifacts/experiment_001'
    """
    if session_dir is None:
        session_dir = f"./artifacts/session_{datetime.now():%Y%m%d_%H%M%S}"

    session_path = Path(session_dir).resolve()
    session_path.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["ARTIFACT_DIR"] = str(session_path)
    return str(session_path)


def get_artifact_session() -> str:
    """Get the current artifact session directory.

    Returns:
        Current session directory (from ARTIFACT_DIR env var or default ./artifacts)
    """
    import os
    return os.getenv("ARTIFACT_DIR", "./artifacts")


def put_artifact(
    obj: Any,
    prefix: str = "obj",
    artifact_dir: Optional[str] = None
) -> str:
    """Store artifact with optimal format (file-based storage).

    Args:
        obj: The artifact to store
        prefix: Type prefix (cgm, metrics, etc) for validation and naming
        artifact_dir: Directory for artifacts (default: ARTIFACT_DIR env var or ./artifacts)

    Returns:
        File path as artifact ID (e.g., "./artifacts/cgm_abc123.parquet")

    Raises:
        ArtifactTypeMismatchError: If prefix is registered and object type doesn't match

    Examples:
        >>> # Use current session
        >>> cgm_aid = put_artifact(df, "cgm")
        './artifacts/cgm_abc123.parquet'

        >>> # Override directory for this artifact
        >>> special_aid = put_artifact(df, "cgm", artifact_dir="./special")
        './special/cgm_def456.parquet'

    Note:
        - Returns a file path, not an abstract ID
        - Optimal formats: Parquet for DataFrames, JSON for dicts
        - Session management via the ARTIFACT_DIR env var
    """
    import os

    # Resolve directory: explicit > env var > default
    if artifact_dir is None:
        artifact_dir = os.getenv("ARTIFACT_DIR", "./artifacts")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Validate type if prefix registered
    if prefix in TYPE_REGISTRY:
        expected = TYPE_REGISTRY[prefix]
        if not isinstance(obj, expected):
            type_names = " | ".join(t.__name__ for t in expected)
            raise ArtifactTypeMismatchError(
                f"Expected {type_names} for prefix '{prefix}', got {type(obj).__name__}"
            )

    # Generate unique ID
    aid = f"{prefix}_{uuid.uuid4().hex[:12]}"

    # Save with optimal format for the object type
    if isinstance(obj, pd.DataFrame):
        path = artifact_dir / f"{aid}.parquet"
        obj.to_parquet(path)
    elif isinstance(obj, dict):
        path = artifact_dir / f"{aid}.json"
        path.write_text(json.dumps(obj, indent=2))
    elif isinstance(obj, list):
        path = artifact_dir / f"{aid}.json"
        path.write_text(json.dumps(obj, indent=2))
    elif isinstance(obj, str):
        path = artifact_dir / f"{aid}.txt"
        path.write_text(obj)
    elif Figure and isinstance(obj, Figure):
        path = artifact_dir / f"{aid}.png"
        obj.savefig(path, dpi=150)
    elif isinstance(obj, io.BytesIO):
        path = artifact_dir / f"{aid}.png"  # Assume image for BytesIO
        with open(path, 'wb') as f:
            f.write(obj.getvalue())
    else:
        path = artifact_dir / f"{aid}.pkl"
        path.write_bytes(pickle.dumps(obj))

    return str(path)


def get_artifact(
    aid: str,
    expected_prefix: Optional[str] = None,
    copy: bool = False
) -> Any:
    """Retrieve artifact from file path.

    Args:
        aid: File path (artifact ID)
        expected_prefix: Validate prefix matches
        copy: Return copy to prevent mutations

    Returns:
        The loaded artifact object

    Raises:
        ArtifactNotFoundError: If artifact file not found
        ArtifactTypeMismatchError: If prefix doesn't match expected

    Examples:
        >>> df = get_artifact(cgm_aid)
        >>> df = get_artifact(cgm_aid, expected_prefix="cgm")  # With validation

    Note:
        - Loads from a file path, not an abstract ID
        - Extension determines format (parquet, json, txt, pkl, png)
    """
    path = Path(aid)

    if not path.exists():
        raise ArtifactNotFoundError(f"Artifact not found: {aid}")

    # Validate prefix if requested
    if expected_prefix:
        actual_prefix = path.stem.split("_")[0]
        if actual_prefix != expected_prefix:
            raise ArtifactTypeMismatchError(
                f"Expected prefix '{expected_prefix}', got '{actual_prefix}'"
            )

    # Load based on extension
    ext = path.suffix
    if ext == ".parquet":
        obj = pd.read_parquet(path)
    elif ext == ".json":
        obj = json.loads(path.read_text())
    elif ext == ".txt":
        obj = path.read_text()
    elif ext == ".pkl":
        obj = pickle.loads(path.read_bytes())
    elif ext == ".png":
        obj = io.BytesIO(path.read_bytes())
    elif ext == ".csv":
        obj = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return obj.copy() if copy and hasattr(obj, "copy") else obj


def clear_artifacts(
    prefix: Optional[str] = None,
    older_than_hours: Optional[float] = None,
    artifact_dir: Optional[str] = None
) -> int:
    """Clean up artifact files.

    Args:
        prefix: Only delete artifacts with this prefix
        older_than_hours: Only delete artifacts older than this
        artifact_dir: Directory to clean (default: ARTIFACT_DIR env var or ./artifacts)

    Returns:
        Number of artifacts deleted

    Examples:
        >>> clear_artifacts()  # Delete everything in current session
        >>> clear_artifacts(prefix="cgm")  # Delete only CGM artifacts
        >>> clear_artifacts(older_than_hours=24)  # Older than 1 day
    """
    import os
    import time

    # Resolve directory
    if artifact_dir is None:
        artifact_dir = os.getenv("ARTIFACT_DIR", "./artifacts")

    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        return 0

    count = 0
    now = time.time()

    for path in artifact_dir.glob("*"):
        if not path.is_file():
            continue

        if prefix and not path.stem.startswith(f"{prefix}_"):
            continue

        if older_than_hours:
            age_hours = (now - path.stat().st_mtime) / 3600
            if age_hours < older_than_hours:
                continue

        path.unlink()
        count += 1

    return count


def has_artifact(aid: str) -> bool:
    """Check if artifact exists without raising exceptions.

    Args:
        aid: File path (artifact identifier)

    Returns:
        True if artifact exists, False otherwise

    Examples:
        >>> if has_artifact(optional_aid):
        ...     data = get_artifact(optional_aid)
    """
    return Path(aid).exists()


def delete_artifact(aid: str) -> None:
    """Remove artifact file.

    Args:
        aid: File path (artifact identifier)

    Raises:
        FileNotFoundError: If artifact not found

    Examples:
        >>> delete_artifact(temp_aid)
    """
    path = Path(aid)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {aid}")
    path.unlink()


def list_artifacts(
    prefix_filter: Optional[str] = None,
    artifact_dir: Optional[str] = None
) -> list[str]:
    """List artifact file paths, optionally filtered by prefix.

    Args:
        prefix_filter: Only return artifacts with this prefix
        artifact_dir: Directory to scan (default: ARTIFACT_DIR env var or ./artifacts)

    Returns:
        List of artifact file paths

    Examples:
        >>> list_artifacts()
        ['./artifacts/cgm_a1b2c3d4e5f6.parquet', './artifacts/metrics_g7h8i9j0k1l2.json']
        >>> list_artifacts(prefix_filter="cgm")
        ['./artifacts/cgm_a1b2c3d4e5f6.parquet']
    """
    import os

    # Resolve directory
    if artifact_dir is None:
        artifact_dir = os.getenv("ARTIFACT_DIR", "./artifacts")

    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        return []

    artifacts = []
    for path in artifact_dir.glob("*"):
        if not path.is_file():
            continue

        if prefix_filter and not path.stem.startswith(f"{prefix_filter}_"):
            continue

        artifacts.append(str(path))

    return artifacts


def artifact_info(aid: str) -> ArtifactInfo:
    """Get metadata about an artifact file.

    Args:
        aid: File path (artifact identifier)

    Returns:
        ArtifactInfo with metadata

    Raises:
        ArtifactNotFoundError: If artifact not found

    Examples:
        >>> info = artifact_info(cgm_aid)
        >>> print(f"Size: {info.size_bytes / 1024:.1f} KB")
    """
    from datetime import datetime, timezone

    path = Path(aid)
    if not path.exists():
        raise ArtifactNotFoundError(f"Artifact not found: {aid}")

    prefix = path.stem.split("_", 1)[0]
    size_bytes = path.stat().st_size
    created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    # Infer type from extension
    ext = path.suffix
    type_name = {
        ".parquet": "DataFrame",
        ".json": "json",
        ".txt": "str",
        ".pkl": "object",
        ".png": "BytesIO"
    }.get(ext, "unknown")

    return ArtifactInfo(
        aid=aid,
        prefix=prefix,
        type_name=type_name,
        size_bytes=size_bytes,
        created_at=created_at,
    )


def total_size(
    prefix_filter: Optional[str] = None,
    artifact_dir: Optional[str] = None
) -> int:
    """Calculate total size of artifacts in bytes.

    Args:
        prefix_filter: Only count artifacts with this prefix
        artifact_dir: Directory to scan (default: ARTIFACT_DIR env var or ./artifacts)

    Returns:
        Total size in bytes

    Examples:
        >>> size_mb = total_size('cgm') / 1024 / 1024
        >>> print(f"CGM artifacts: {size_mb:.1f} MB")
    """
    aids = list_artifacts(prefix_filter, artifact_dir)
    total = 0
    for aid in aids:
        try:
            info = artifact_info(aid)
            if info.size_bytes:
                total += info.size_bytes
        except (ArtifactNotFoundError, FileNotFoundError):
            # Artifact was deleted between list and info call
            pass
    return total


# Public API
__all__ = [
    # Core functions
    "put_artifact",
    "get_artifact",
    "has_artifact",
    "delete_artifact",
    "list_artifacts",
    "artifact_info",
    "clear_artifacts",
    "register_type",
    "total_size",
    # Session management (new in file-based version)
    "set_artifact_session",
    "get_artifact_session",
    # Models and types
    "ArtifactInfo",
    # Custom exceptions
    "ArtifactNotFoundError",
    "ArtifactTypeMismatchError",
    # Internal (for inspection/testing)
    "TYPE_REGISTRY",
]
