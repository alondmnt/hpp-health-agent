"""
Column contract for diet logging data QC.

Reports which columns are present and which are missing. Validates structure
only - missing data is flagged by the rule groups, never filled here.
"""


def validate_required_columns(df_columns: list) -> tuple[bool, list]:
    """
    Validate that required columns are present.

    Note: participant_uuid/participant_id are optional. If neither is present,
    the tool creates a synthetic ID internally for grouping and strips it from
    the output.

    Args:
        df_columns: List of column names in DataFrame

    Returns:
        Tuple of (is_valid, missing_required_columns)
    """
    required_cols = ['local_timestamp']

    missing = []
    for col in required_cols:
        if col not in df_columns:
            missing.append(col)

    return (len(missing) == 0, missing)


def report_optional_columns(df_columns: list) -> tuple[list, list]:
    """
    Report which optional columns are present and missing.

    Args:
        df_columns: List of column names in DataFrame

    Returns:
        Tuple of (present_columns, missing_columns)
    """
    # Grouping matches the completeness rule: weight_g is treated as a core
    # macro there, so it is listed as one here too.
    optional_cols = {
        'Core nutrients': ['calories_kcal', 'carbohydrate_g', 'lipid_g', 'protein_g', 'weight_g'],
        'Extended nutrients': ['sodium_mg', 'alcohol_g', 'dietary_fiber_g'],
        'Text fields': ['short_food_name', 'product_name', 'food_category'],
        'IDs': ['food_id']
    }

    present = []
    missing = []

    for category, cols in optional_cols.items():
        for col in cols:
            if col in df_columns:
                present.append(f"{col} ({category})")
            else:
                missing.append(f"{col} ({category})")

    return present, missing
