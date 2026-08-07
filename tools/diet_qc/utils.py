"""
Shared utility functions for QC pipeline.

Consolidates common helper methods to reduce code duplication.
"""

import logging
import pandas as pd
from typing import Optional, Tuple


# Constants
_SYNTHETIC_PARTICIPANT_ID = "synthetic_participant"
_SUPPORTED_PARTICIPANT_COLUMNS = ['participant_id', 'participant_uuid']


def prepare_dataframe_for_qc(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, bool]:
    """
    Prepare DataFrame for QC processing by handling participant ID detection.
    
    Handles three cases:
    1. DataFrame with MultiIndex → reset to columns
    2. Explicit participant ID column (participant_id or participant_uuid) → use it
    3. No participant ID → create synthetic ID for internal grouping
    
    Args:
        df: Input DataFrame (may have MultiIndex or regular columns)
    
    Returns:
        Tuple of:
        - df: DataFrame with participant column guaranteed
        - participant_col: Name of participant column to use
        - synthetic_used: True if we created a synthetic ID
    
    Example:
        >>> df, participant_col, synthetic = prepare_dataframe_for_qc(raw_df)
        >>> # Process using participant_col...
        >>> if synthetic:
        >>>     df = df.drop(columns=['_participant_id'])  # Remove synthetic ID from output
    """
    # Step 1: Handle MultiIndex
    if df.index.nlevels > 1:
        logging.info("Detected MultiIndex with %d levels, resetting to columns", df.index.nlevels)
        df = df.reset_index()
    
    # Step 2: Detect participant ID column
    if 'participant_id' in df.columns:
        logging.info("Using existing 'participant_id' column")
        return df.copy(), 'participant_id', False
    elif 'participant_uuid' in df.columns:
        logging.info("Using existing 'participant_uuid' column")
        return df.copy(), 'participant_uuid', False
    else:
        logging.info("No participant ID found, creating synthetic ID: %s", _SYNTHETIC_PARTICIPANT_ID)
        df_copy = df.copy()
        df_copy['_participant_id'] = _SYNTHETIC_PARTICIPANT_ID
        return df_copy, '_participant_id', True


def check_columns_exist(df: pd.DataFrame, columns: list) -> tuple[list, list]:
    """
    Check which columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        columns: List of column names to check
    
    Returns:
        Tuple of (present_columns, missing_columns)
    """
    present = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    return present, missing


def get_participant_col(df: pd.DataFrame) -> Optional[str]:
    """
    Get participant column name (handles uuid vs id variants).
    
    Args:
        df: DataFrame to check
    
    Returns:
        Column name or None if not found
    """
    if 'participant_uuid' in df.columns:
        return 'participant_uuid'
    elif 'participant_id' in df.columns:
        return 'participant_id'
    return None
