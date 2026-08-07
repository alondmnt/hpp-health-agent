"""
Completeness validation rules.

Checks for missing nutrient data and food name fields.
- any_core_macro_missing: FAIL severity (any one of 5 core macros missing)
- all_food_names_missing: FAIL severity (all 3 food name fields missing)
- any_other_nutrient_missing: CHECK severity (any other nutrient field missing)
"""

import logging
import pandas as pd
from typing import Dict, List
from ..utils import check_columns_exist


def validate_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate completeness of nutrient and food name fields.
    
    Rules:
    - any_core_macro_missing: ANY of (calories, carbs, lipid, protein, weight) is NA → FAIL
    - all_food_names_missing: ALL 3 (short_food_name, product_name, food_category) are NA → FAIL
    - any_other_nutrient_missing: Any of (sodium, alcohol, fiber) is missing → CHECK
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with qc_completeness column containing flags
    """
    result_df = df.copy()
    result_df['qc_completeness'] = ''
    
    # Core macros (for FAIL severity check)
    core_macros = ['calories_kcal', 'carbohydrate_g', 'lipid_g', 'protein_g', 'weight_g']
    
    # Food name fields (for FAIL severity check)
    food_name_fields = ['short_food_name', 'product_name', 'food_category']
    
    # Other nutrients (for CHECK severity)
    other_nutrients = ['sodium_mg', 'alcohol_g', 'dietary_fiber_g']
    
    # Check which columns exist using helper
    available_core_macros, _ = check_columns_exist(df, core_macros)
    available_food_names, _ = check_columns_exist(df, food_name_fields)
    available_other_nutrients, _ = check_columns_exist(df, other_nutrients)
    
    if not available_core_macros:
        logging.warning("Completeness check: No core macro columns found, skipping macro checks")
    
    if not available_food_names:
        logging.warning("Completeness check: No food name columns found, skipping food name checks")
    
    # Vectorized completeness checks
    flags_list = []
    
    # Check 1: If ANY core macro is missing → FAIL
    if available_core_macros:
        missing_core_mask = result_df[available_core_macros].isna()
        missing_core_cols = missing_core_mask.apply(
            lambda row: ','.join(row.index[row].tolist()) if row.any() else '', axis=1
        )
        core_flags = missing_core_cols.apply(
            lambda x: f'any_core_macro_missing:{x}' if x else ''
        )
        flags_list.append(core_flags)
    
    # Check 2: If ALL food names are missing → FAIL
    if available_food_names:
        all_food_missing = (result_df[available_food_names].isna() | 
                           (result_df[available_food_names] == '')).all(axis=1)
        food_flags = all_food_missing.apply(lambda x: 'all_food_names_missing' if x else '')
        flags_list.append(food_flags)
    
    # Check 3: If any other nutrient is missing → CHECK
    if available_other_nutrients:
        missing_other_mask = result_df[available_other_nutrients].isna()
        missing_other_cols = missing_other_mask.apply(
            lambda row: ','.join(row.index[row].tolist()) if row.any() else '', axis=1
        )
        other_flags = missing_other_cols.apply(
            lambda x: f'any_other_nutrient_missing:{x}' if x else ''
        )
        flags_list.append(other_flags)
    
    # Combine all flags with '|' separator
    if flags_list:
        result_df['qc_completeness'] = pd.DataFrame(flags_list).T.apply(
            lambda row: '|'.join([x for x in row if x]), axis=1
        )
    else:
        result_df['qc_completeness'] = ''
    
    flagged_count = (result_df['qc_completeness'] != '').sum()
    logging.info(f"Completeness: {flagged_count} rows flagged")
    
    return result_df

