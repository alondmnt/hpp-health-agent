"""
Severity roll-up.

Aggregates the four rule-group flag columns into a single per-row severity.
Deterministic: no LLM involvement.

Severity hierarchy (3 levels):
- fail:  critical data quality issues
- check: warning flags worth a human look
- pass:  no violations
"""

import logging
import pandas as pd


# Flags that escalate a row to 'fail'
_FAIL_COMPLETENESS_PATTERNS = 'any_core_macro_missing|all_food_names_missing'
_FAIL_INTEGRITY_PATTERNS = 'participant_low_cal_period|before_study_start'

# Flags that raise a row to 'check' (unless a fail-level flag also applies)
_CHECK_COMPLETENESS_PATTERNS = 'any_other_nutrient_missing'
_CHECK_INTEGRITY_PATTERNS = 'timezone|first_day|day_low_cal|day_low_items|participant_low_days'

_RULE_GROUPS = ['completeness', 'range', 'consistency', 'integrity']


def calculate_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall QC severity from the individual rule-group flags.

    Any range or consistency flag is a fail. Completeness and integrity flags
    split between fail and check depending on which rule fired.

    Args:
        df: DataFrame with qc_completeness, qc_range, qc_consistency,
            qc_integrity columns (each may be absent)

    Returns:
        DataFrame with qc_severity and qc_failed_groups columns added
    """
    result_df = df.copy()
    result_df['qc_severity'] = 'pass'
    result_df['qc_failed_groups'] = ''

    present = {g: f'qc_{g}' in df.columns for g in _RULE_GROUPS}

    # Build the comma-joined list of rule groups that flagged each row
    failed_groups_parts = []
    for group in _RULE_GROUPS:
        if not present[group]:
            continue
        col = f'qc_{group}'
        flagged = result_df[col].notna() & (result_df[col] != '')
        failed_groups_parts.append(flagged.apply(lambda x, g=group: g if x else ''))

    if failed_groups_parts:
        result_df['qc_failed_groups'] = pd.DataFrame(failed_groups_parts).T.apply(
            lambda row: ','.join([x for x in row if x]), axis=1
        )

    # CHECK severity conditions
    check_mask = pd.Series(False, index=result_df.index)

    if present['completeness']:
        check_mask |= result_df['qc_completeness'].str.contains(
            _CHECK_COMPLETENESS_PATTERNS, na=False
        )

    if present['integrity']:
        integrity_has_check = result_df['qc_integrity'].str.contains(
            _CHECK_INTEGRITY_PATTERNS, regex=True, na=False
        )
        # Exclude rows that also carry a fail-level integrity flag
        integrity_has_fail = result_df['qc_integrity'].str.contains(
            _FAIL_INTEGRITY_PATTERNS, regex=True, na=False
        )
        check_mask |= integrity_has_check & ~integrity_has_fail

    result_df.loc[check_mask, 'qc_severity'] = 'check'

    # FAIL severity conditions (overwrite check)
    fail_mask = pd.Series(False, index=result_df.index)

    if present['completeness']:
        fail_mask |= result_df['qc_completeness'].str.contains(
            _FAIL_COMPLETENESS_PATTERNS, regex=True, na=False
        )

    if present['range']:
        fail_mask |= result_df['qc_range'].notna() & (result_df['qc_range'] != '')

    if present['consistency']:
        fail_mask |= result_df['qc_consistency'].notna() & (result_df['qc_consistency'] != '')

    if present['integrity']:
        fail_mask |= result_df['qc_integrity'].str.contains(
            _FAIL_INTEGRITY_PATTERNS, regex=True, na=False
        )

    result_df.loc[fail_mask, 'qc_severity'] = 'fail'

    severity_counts = result_df['qc_severity'].value_counts()
    logging.info("Severity calculated:")
    for severity, count in severity_counts.items():
        logging.info(f"  - {severity}: {count}")

    return result_df
