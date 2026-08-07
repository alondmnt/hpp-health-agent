"""Utility functions for file hashing, time conversions, and timezone handling.

This module provides helper functions for computing file checksums,
converting frequency strings to minutes, and handling timezone localization
for CGM data processing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import pandas as pd

DEFAULT_TIMEZONE = "UTC"


def localize_timestamps(
    dt_obj: Union[pd.Series, pd.DatetimeIndex],
    tz: str = DEFAULT_TIMEZONE
) -> Union[pd.Series, pd.DatetimeIndex]:
    """Normalize timestamps to a common timezone for cross-sensor alignment.

    Ensures data from different sources (CGM, diet, wearables) can be
    jointly analyzed by converting all timestamps to the same timezone.

    Rule:
    - Naive timestamps → localize to tz (assume local time)
    - Aware timestamps → convert to tz (align to common timezone)

    DST edge cases:
    - nonexistent times (spring forward): shift forward to valid time
    - ambiguous times (fall back): use standard time (keeps data)

    Args:
        dt_obj: Datetime Series or DatetimeIndex (naive or timezone-aware).
        tz: Target timezone (e.g., "UTC", "America/New_York").

    Returns:
        Timezone-aware Series or DatetimeIndex, all in the specified timezone.
    """
    if isinstance(dt_obj, pd.DatetimeIndex):
        if dt_obj.tz is None:
            return dt_obj.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
        return dt_obj.tz_convert(tz)
    else:  # Series
        # pd.to_datetime may return object dtype when timestamps have mixed
        # timezone offsets (e.g. "+02:00" and "+03:00" across a DST boundary).
        # Convert to UTC first to get a proper datetime64 dtype, then convert.
        if dt_obj.dtype == object:
            dt_obj = pd.to_datetime(dt_obj, utc=True)
        if dt_obj.dt.tz is None:
            return dt_obj.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
        return dt_obj.dt.tz_convert(tz)


def sha256_file(path: str) -> Optional[str]:
    """Compute SHA256 checksum of a file.
    
    Args:
        path: Path to the file.
        
    Returns:
        Hexadecimal digest string, or None if file doesn't exist.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def minutes_from_freq_str(freq: str) -> float:
    """Convert a frequency string to minutes.
    
    Supports common time unit suffixes: 'min'/'m' (minutes), 's' (seconds), 'h' (hours).
    
    Args:
        freq: Frequency string (e.g., "5min", "30s", "1h").
        
    Returns:
        Time value in minutes.
        
    Raises:
        ValueError: If the frequency string format is not recognized.
        
    Examples:
        >>> minutes_from_freq_str("5min")
        5.0
        >>> minutes_from_freq_str("30s")
        0.5
        >>> minutes_from_freq_str("2h")
        120.0
    """
    units = freq.lower()
    if units.endswith("min"):
        return float(units.replace("min", ""))
    if units.endswith("m"):
        return float(units[:-1])
    if units.endswith("s"):
        return float(units[:-1]) / 60.0
    if units.endswith("h"):
        return float(units[:-1]) * 60.0
    raise ValueError(f"Unsupported frequency: {freq}")


