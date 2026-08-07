"""
Data integrity validation rules.

Checks for data quality issues and temporal patterns.
Severity: FAIL (before_study_start, participant_low_cal_period) or CHECK
(timezone, first_day, day-level rules, participant_low_days)

## A note on the thresholds below

Two of these rules encode *study design*, not physiology, so they are disabled
by default and must be switched on deliberately:

- STUDY_START_DATE: in a real deployment this is the date data collection
  opened, and anything earlier is a logging error. There is no universal value,
  so the default is None (rule off).
- EXPECTED_UTC_OFFSETS: the UTC offsets participants are expected to log in.
  Anything else suggests a device or timezone misconfiguration. Again
  cohort-specific, so the default is None (rule off).

The remaining four thresholds are operational judgement calls about what
constitutes a usable logging day or participant. They are round numbers chosen
for plausibility, not fitted to any dataset, and callers should tune them.
"""

import logging
import pandas as pd
from datetime import date
from typing import Optional, Set
from ..utils import get_participant_col

# Study-design rules: disabled unless configured (see module docstring)
STUDY_START_DATE: Optional[date] = None
EXPECTED_UTC_OFFSETS: Optional[Set[int]] = None

# Operational thresholds: plausibility judgements, not fitted values
_MIN_DAY_CALORIES = 500
_MIN_DAY_ITEMS = 3
_MIN_PARTICIPANT_DAYS = 5
_MIN_PARTICIPANT_TOTAL_CALORIES = 1000


def validate_data_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data integrity including timestamps and aggregations.
    
    Rules (Severity):
    - timezone: UTC offset outside EXPECTED_UTC_OFFSETS → CHECK (off by default)
    - before_study_start: Before STUDY_START_DATE → FAIL (off by default)
    - day_low_cal: Days < 500 kcal total → CHECK
    - day_low_items: Days < 3 items → CHECK
    - first_day: First logging day per participant → CHECK
    - participant_low_days: Participants < 5 full days → CHECK
    - participant_low_cal_period: Participants < 1000 kcal total period → FAIL
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with qc_integrity column containing flags
    """
    result_df = df.copy()
    
    # Check for required columns
    participant_col = get_participant_col(df)
    has_participant = participant_col is not None
    has_timestamp = 'local_timestamp' in df.columns
    has_calories = 'calories_kcal' in df.columns
    
    # Initialize flags as empty lists for each row (will be joined at the end)
    flags_list = [[] for _ in range(len(result_df))]
    
    # Parse timestamps once upfront
    if has_timestamp:
        result_df['_ts'] = pd.to_datetime(result_df['local_timestamp'], errors='coerce')
        result_df['_date'] = result_df['_ts'].dt.date
        
        # 1. Timezone check (vectorized)
        _apply_timezone_flags(result_df, flags_list)
        
        # 2. Study-start check (vectorized)
        _apply_study_start_flags(result_df, flags_list)
    
    # Participant and day-level checks
    if has_participant and has_timestamp:
        try:
            # Calculate daily statistics
            daily_stats = _compute_daily_stats(result_df, participant_col, has_calories)
            
            # Calculate participant statistics
            participant_stats = _compute_participant_stats(result_df, participant_col, has_calories)
            
            # Find first logging day per participant (vectorized)
            first_days = result_df.groupby(participant_col)['_date'].transform('min')
            
            # Mark first row per participant+day for day-level flags
            result_df['_is_first_row_of_day'] = (
                result_df.groupby([participant_col, '_date']).cumcount() == 0
            )
            
            # Merge daily stats back to rows
            result_df = result_df.merge(
                daily_stats,
                left_on=[participant_col, '_date'],
                right_index=True,
                how='left',
                suffixes=('', '_daily')
            )
            
            # Merge participant stats back to rows
            result_df = result_df.merge(
                participant_stats,
                left_on=participant_col,
                right_index=True,
                how='left',
                suffixes=('', '_participant')
            )
            
            # Store first_days in DataFrame for vectorized access
            result_df['_first_day'] = first_days.values
            
            # 3. Day-level checks (vectorized)
            _apply_day_level_flags(result_df, flags_list, has_calories)
            
            # 4. Participant-level checks (vectorized)
            _apply_participant_level_flags(result_df, flags_list, has_calories)
            
        except Exception as e:
            logging.warning(f"Integrity check: Could not calculate aggregations: {e}")
    
    # Join all flags into the qc_integrity column
    result_df['qc_integrity'] = ['|'.join(f) if f else '' for f in flags_list]
    
    # Clean up temporary columns
    temp_cols = [c for c in result_df.columns if c.startswith('_')]
    result_df = result_df.drop(columns=temp_cols, errors='ignore')
    
    flagged_count = (result_df['qc_integrity'] != '').sum()
    logging.info(f"Integrity checks: {flagged_count} rows flagged")
    
    return result_df


def _compute_daily_stats(df: pd.DataFrame, participant_col: str, has_calories: bool) -> pd.DataFrame:
    """Compute daily statistics per participant."""
    if has_calories:
        daily_stats = df.groupby([participant_col, '_date']).agg(
            _daily_calories=('calories_kcal', 'sum'),
            _daily_item_count=('calories_kcal', 'count')
        )
    else:
        daily_stats = df.groupby([participant_col, '_date']).size().to_frame('_daily_item_count')
    return daily_stats


def _compute_participant_stats(df: pd.DataFrame, participant_col: str, has_calories: bool) -> pd.DataFrame:
    """Compute participant-level statistics."""
    participant_days = df.groupby(participant_col)['_date'].nunique()
    if has_calories:
        participant_total_cal = df.groupby(participant_col)['calories_kcal'].sum()
        return pd.DataFrame({
            '_total_days': participant_days,
            '_total_calories': participant_total_cal
        })
    return pd.DataFrame({'_total_days': participant_days})


def _apply_timezone_flags(df: pd.DataFrame, flags_list: list) -> None:
    """Flag rows whose UTC offset is outside the expected set.

    No-op unless EXPECTED_UTC_OFFSETS is configured (see module docstring).
    """
    if not EXPECTED_UTC_OFFSETS:
        return

    ts_col = df['_ts']
    
    # Check rows that have timezone info
    has_tz = ts_col.apply(lambda x: hasattr(x, 'tzinfo') and x.tzinfo is not None if pd.notna(x) else False)
    
    if has_tz.any():
        # Get offset hours for rows with timezone
        offset_hours = ts_col[has_tz].apply(lambda x: x.utcoffset().total_seconds() / 3600)
        invalid_tz_mask = ~offset_hours.isin(EXPECTED_UTC_OFFSETS)
        
        for idx in offset_hours[invalid_tz_mask].index:
            flags_list[df.index.get_loc(idx)].append(f'timezone:offset={offset_hours[idx]}h')


def _apply_study_start_flags(df: pd.DataFrame, flags_list: list) -> None:
    """Flag rows dated before the study start date.

    No-op unless STUDY_START_DATE is configured (see module docstring).
    """
    if STUDY_START_DATE is None:
        return

    too_early_mask = df['_date'] < STUDY_START_DATE

    for idx in df[too_early_mask].index:
        date_val = df.at[idx, '_date']
        flags_list[df.index.get_loc(idx)].append(f'before_study_start:{date_val}')


def _apply_day_level_flags(df: pd.DataFrame, flags_list: list, has_calories: bool) -> None:
    """Apply day-level validation flags (vectorized)."""
    # First day flag (applies to ALL items on first day)
    first_day_mask = df['_date'] == df['_first_day']
    for idx in df[first_day_mask].index:
        flags_list[df.index.get_loc(idx)].append('first_day')
    
    # Day-level stats flags (only for first row of each day)
    first_row_mask = df['_is_first_row_of_day']
    
    # Low calorie day
    if has_calories and '_daily_calories' in df.columns:
        low_cal_mask = first_row_mask & (df['_daily_calories'] < _MIN_DAY_CALORIES)
        for idx in df[low_cal_mask].index:
            cal_val = df.at[idx, '_daily_calories']
            flags_list[df.index.get_loc(idx)].append(f'day_low_cal:{cal_val:.0f}kcal')
    
    # Low item count day
    if '_daily_item_count' in df.columns:
        low_items_mask = first_row_mask & (df['_daily_item_count'] < _MIN_DAY_ITEMS)
        for idx in df[low_items_mask].index:
            item_count = df.at[idx, '_daily_item_count']
            flags_list[df.index.get_loc(idx)].append(f'day_low_items:{int(item_count)}items')


def _apply_participant_level_flags(df: pd.DataFrame, flags_list: list, has_calories: bool) -> None:
    """Apply participant-level validation flags (vectorized)."""
    # Low days
    if '_total_days' in df.columns:
        low_days_mask = df['_total_days'] < _MIN_PARTICIPANT_DAYS
        for idx in df[low_days_mask].index:
            days_val = df.at[idx, '_total_days']
            flags_list[df.index.get_loc(idx)].append(f'participant_low_days:{int(days_val)}days')
    
    # Low calories over entire period
    if has_calories and '_total_calories' in df.columns:
        low_cal_mask = df['_total_calories'] < _MIN_PARTICIPANT_TOTAL_CALORIES
        for idx in df[low_cal_mask].index:
            cal_val = df.at[idx, '_total_calories']
            flags_list[df.index.get_loc(idx)].append(f'participant_low_cal_period:{cal_val:.0f}kcal')

