"""
Range check validation rules.

Checks for values outside acceptable ranges. Severity: FAIL
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils import check_columns_exist, get_participant_col


def validate_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate nutrient values are within acceptable ranges.
    
    Rules:
    - non_negative: Negative nutrients (calories, carbs, lipid, protein, sodium, alcohol, fiber)
    - weight_bounds: 0 < weight_g < 2000
    - calorie_day_upper: Days > 5000 kcal total
    - nutrient_weight_ratio: Nutrients <= total weight (±10%)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with qc_range column containing flags
    """
    result_df = df.copy()
    result_df['qc_range'] = ''
    
    # Define nutrient columns to check
    nutrient_columns = {
        'calories_kcal', 'carbohydrate_g', 'lipid_g', 'protein_g',
        'sodium_mg', 'alcohol_g', 'dietary_fiber_g'
    }
    
    # Define macronutrient columns for mass calculations (excludes calories)
    mass_nutrient_columns = {
        'carbohydrate_g', 'lipid_g', 'protein_g',
        'sodium_mg', 'alcohol_g', 'dietary_fiber_g'
    }
    
    # Check which columns exist using helpers
    available_nutrients, _ = check_columns_exist(df, list(nutrient_columns))
    available_mass_nutrients, _ = check_columns_exist(df, list(mass_nutrient_columns))
    weight_col = 'weight_g' if 'weight_g' in df.columns else None
    
    # Prepare for day-level calculations
    participant_col = get_participant_col(df)
    has_participant = participant_col is not None
    has_timestamp = 'local_timestamp' in df.columns
    
    # Calculate daily calories if possible
    daily_calories = None
    if has_participant and has_timestamp and 'calories_kcal' in df.columns:
        try:
            temp_df = result_df.copy()
            temp_df['date'] = pd.to_datetime(temp_df['local_timestamp']).dt.date
            daily_calories = temp_df.groupby([participant_col, 'date'])['calories_kcal'].sum()
        except Exception as e:
            logging.warning(f"Range checks: Could not calculate daily calories: {e}")
    
    # Vectorized range checks
    flags_list = []
    
    # 1. Check for negative nutrients
    if available_nutrients:
        negative_mask = result_df[available_nutrients] < 0
        negative_cols = negative_mask.apply(
            lambda row: ','.join(row.index[row].tolist()) if row.any() else '', axis=1
        )
        negative_flags = negative_cols.apply(
            lambda x: f'non_negative:{x}' if x else ''
        )
        flags_list.append(negative_flags)
    
    # 2. Weight bounds check
    if weight_col:
        weight_invalid = (result_df[weight_col] <= 0) | (result_df[weight_col] >= 2000)
        weight_flags = result_df[weight_col].where(weight_invalid).apply(
            lambda x: f'weight_bounds:{x}g' if pd.notna(x) else ''
        )
        flags_list.append(weight_flags)
    
    # 3. Daily calorie upper bound (checked once per day)
    if daily_calories is not None and has_participant and has_timestamp:
        try:
            # Mark first row of each day that exceeds 5000 kcal
            days_over = daily_calories[daily_calories > 5000]
            calorie_flags = pd.Series('', index=result_df.index)
            
            for (participant, date), day_total in days_over.items():
                mask = (temp_df[participant_col] == participant) & (temp_df['date'] == date)
                if mask.any():
                    first_idx = temp_df[mask].index[0]
                    calorie_flags.loc[first_idx] = f'calorie_day_upper:{day_total:.0f}kcal'
            
            flags_list.append(calorie_flags)
        except Exception as e:
            logging.warning(f"Range checks: Could not flag daily calories: {e}")
    
    # 4. Nutrient to weight ratio
    if weight_col and available_mass_nutrients:
        # Convert sodium from mg to g and sum all mass nutrients
        mass_df = result_df[available_mass_nutrients].copy()
        if 'sodium_mg' in mass_df.columns:
            mass_df['sodium_mg'] = mass_df['sodium_mg'] / 1000
        
        total_nutrients = mass_df.sum(axis=1)
        weight_values = result_df[weight_col]
        
        # Check ratio with 10% tolerance
        ratio_invalid = (weight_values > 0) & (total_nutrients > weight_values * 1.1)
        ratio_flags = pd.Series('', index=result_df.index)
        ratio_flags[ratio_invalid] = (
            'nutrient_weight_ratio:' + 
            total_nutrients[ratio_invalid].round(1).astype(str) + 'g>' +
            weight_values[ratio_invalid].astype(str) + 'g'
        )
        flags_list.append(ratio_flags)
    
    # Combine all flags with '|' separator
    if flags_list:
        result_df['qc_range'] = pd.DataFrame(flags_list).T.apply(
            lambda row: '|'.join([x for x in row if x]), axis=1
        )
    else:
        result_df['qc_range'] = ''
    
    flagged_count = (result_df['qc_range'] != '').sum()
    logging.info(f"Range checks: {flagged_count} rows flagged")
    
    return result_df

