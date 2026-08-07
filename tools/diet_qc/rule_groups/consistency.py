"""
Consistency validation rules.

Checks for internal consistency and duplicates. Severity: FAIL
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils import get_participant_col, check_columns_exist


def validate_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate internal consistency of data.
    
    Rules:
    - duplicate_logging: Same participant, timestamp, and food name (flags duplicates only, not first occurrence)
    - macro_equation: calories ≈ 4×protein + 4×carbs + 9×lipid + 7×alcohol (±15%)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with qc_consistency column containing flags
    """
    result_df = df.copy()
    result_df['qc_consistency'] = ''
    
    # Check for required columns using helper
    participant_col = get_participant_col(df)
    has_participant = participant_col is not None
    has_timestamp = 'local_timestamp' in df.columns
    
    # Create a composite food name field using priority: short_food_name > product_name > food_category
    result_df['_food_name_for_dup_check'] = result_df.apply(
        lambda row: (row.get('short_food_name') if pd.notna(row.get('short_food_name')) and row.get('short_food_name') != ''
                     else row.get('product_name') if pd.notna(row.get('product_name')) and row.get('product_name') != ''
                     else row.get('food_category') if pd.notna(row.get('food_category')) and row.get('food_category') != ''
                     else None),
        axis=1
    )
    
    # Check for duplicates if we have necessary columns
    duplicate_mask = pd.Series([False] * len(df), index=df.index)
    if has_participant and has_timestamp:
        # Find duplicates (keep='first' means only flag duplicates, not the first occurrence)
        duplicate_mask = result_df.duplicated(
            subset=[participant_col, 'local_timestamp', '_food_name_for_dup_check'],
            keep='first'
        )
    
    # Check which macro columns exist for equation
    macro_cols = {
        'calories_kcal': 1,
        'protein_g': 4,
        'carbohydrate_g': 4,
        'lipid_g': 9,
        'alcohol_g': 7
    }
    available_macros = {col: multiplier for col, multiplier in macro_cols.items() if col in df.columns}
    
    # Vectorized consistency checks
    flags_list = []
    
    # 1. Duplicate logging check (already vectorized)
    duplicate_flags = duplicate_mask.apply(lambda x: 'duplicate_logging' if x else '')
    flags_list.append(duplicate_flags)
    
    # 2. Macro equation check (vectorized)
    if 'calories_kcal' in available_macros:
        calories = result_df['calories_kcal']
        
        # Calculate expected calories from macros
        calculated_calories = pd.Series(0.0, index=result_df.index)
        has_data = pd.Series(False, index=result_df.index)
        
        for col, multiplier in available_macros.items():
            if col != 'calories_kcal':
                col_values = result_df[col].fillna(0)
                calculated_calories += col_values * multiplier
                # Track if we have at least some macro data
                if col in ['protein_g', 'carbohydrate_g', 'lipid_g']:
                    has_data |= result_df[col].notna()
        
        # Check equation where we have calories and at least some macro data
        valid_check = calories.notna() & (calories > 0) & ((calculated_calories > 0) | has_data)
        
        # Allow 15% tolerance
        lower_bound = calculated_calories * 0.85
        upper_bound = calculated_calories * 1.15
        
        macro_invalid = valid_check & ((calories < lower_bound) | (calories > upper_bound))
        
        # Build flag strings
        diff = (calories - calculated_calories).abs()
        diff_pct = (diff / calories * 100).where(calories > 0, 0).round(1)
        
        macro_flags = pd.Series('', index=result_df.index)
        macro_flags[macro_invalid] = (
            'macro_equation:reported=' + calories[macro_invalid].round(0).astype(str) + 'kcal,' +
            'calculated=' + calculated_calories[macro_invalid].round(0).astype(str) + 'kcal,' +
            'diff=' + diff_pct[macro_invalid].astype(str) + '%'
        )
        flags_list.append(macro_flags)
    
    # Combine all flags with '|' separator
    if flags_list:
        result_df['qc_consistency'] = pd.DataFrame(flags_list).T.apply(
            lambda row: '|'.join([x for x in row if x]), axis=1
        )
    else:
        result_df['qc_consistency'] = ''
    
    # Drop temporary column
    result_df = result_df.drop(columns=['_food_name_for_dup_check'], errors='ignore')
    
    flagged_count = (result_df['qc_consistency'] != '').sum()
    logging.info(f"Consistency checks: {flagged_count} rows flagged")
    
    return result_df

