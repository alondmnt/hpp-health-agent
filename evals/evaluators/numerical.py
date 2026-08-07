"""Eval verifying that reported CGM metrics match the tool-computed values.

Threshold-based validation; requires ground truth.

Evaluation: Numerical Accuracy

Verify reported CGM metrics match ground truth computations.
Checks accuracy of extracted metrics against known values:
- Critical: cgm_mean, cgm_cv, cgm_in_range_70_180
- Important: cgm_ea1c, cgm_gmi, cgm_adrr, cgm_lbgi, cgm_hbgi

Pass criteria: All critical metrics must be within threshold.
"""

from typing import Dict, Optional
from evals.core.eval_decorator import eval, EvalResult
from evals.core.utils import extract_metric_patterns, calculate_mae


# Default pass thresholds; eval_config.yaml overrides these
# Keys use cgm_* standard names (matching ground_truth.json)
THRESHOLDS = {
    'cgm_mean': 1.0,           # MAE < 1.0 mg/dL
    'cgm_cv': 1.0,             # MAE < 1.0%
    'cgm_in_range_70_180': 1.0,  # MAE < 1.0%
    'cgm_ea1c': 0.1,           # MAE < 0.1
    'cgm_gmi': 0.1,            # MAE < 0.1
    'cgm_adrr': 1.0,           # MAE < 1.0
    'cgm_lbgi': 0.5,           # MAE < 0.5
    'cgm_hbgi': 0.5,           # MAE < 0.5
}

CRITICAL_METRICS = ['cgm_mean', 'cgm_cv', 'cgm_in_range_70_180']
IMPORTANT_METRICS = ['cgm_ea1c', 'cgm_gmi', 'cgm_adrr', 'cgm_lbgi', 'cgm_hbgi']


@eval(
    version="1.0.0",
    categories=["cgm", "numerical", "accuracy"],
    requires_ground_truth=True,
    requires_llm=False,
    description="Check extracted metrics match ground truth within thresholds"
)
def eval_numerical(
    report_text: str,
    ground_truth: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None
) -> EvalResult:
    """
    Evaluate numerical accuracy of metrics in report.

    Args:
        report_text: Full report text
        ground_truth: Dictionary of metric name -> value (required)
        config: Optional config dict with 'thresholds' key

    Returns:
        EvalResult with pass/fail, MAE, and detailed errors
    """
    cfg = config or {}
    thresholds = cfg.get('thresholds', THRESHOLDS)

    if ground_truth is None:
        return EvalResult(
            pass_=False,
            score=0.0,
            summary="Ground truth required but not provided",
            failures=["ground_truth parameter is required for numerical accuracy eval"]
        )
    # Extract metrics from report
    extracted_metrics = extract_metric_patterns(report_text)
    
    # Calculate MAE
    mae_results = calculate_mae(extracted_metrics, ground_truth)
    
    # Check pass criteria - only check metrics we care about
    failures = []
    critical_failures = []
    important_failures = []
    
    # Only evaluate metrics that have thresholds defined
    for metric in thresholds.keys():
        details = mae_results['per_metric'].get(metric, {})

        if details.get('missing'):
            # Metric not found in report
            if metric in CRITICAL_METRICS:
                critical_failures.append(
                    f"{metric}: not found in report (critical)"
                )
            elif metric in IMPORTANT_METRICS:
                important_failures.append(
                    f"{metric}: not found in report (important)"
                )
            else:
                failures.append(f"{metric}: not found in report")
            continue

        error = details.get('error')
        if error is None:
            continue

        threshold = thresholds.get(metric, float('inf'))
        
        if error > threshold:
            msg = f"{metric}: MAE {error:.2f} exceeds threshold {threshold}"
            if metric in CRITICAL_METRICS:
                critical_failures.append(msg + " (critical)")
            elif metric in IMPORTANT_METRICS:
                important_failures.append(msg + " (important)")
            else:
                failures.append(msg)
    
    # Determine pass/fail
    passed = len(critical_failures) == 0  # Must pass all critical metrics
    
    # Generate summary
    # Count only metrics we care about (in thresholds)
    checked_metrics = [m for m in thresholds.keys() if m in mae_results['per_metric']]
    extracted_checked = sum(1 for m in checked_metrics
                           if not mae_results['per_metric'][m].get('missing'))

    overall_mae = mae_results['overall_mae']

    if overall_mae is not None:
        summary = f"Extracted {extracted_checked}/{len(thresholds)} key metrics, overall MAE: {overall_mae:.3f}"
    else:
        summary = f"Extracted {extracted_checked}/{len(thresholds)} key metrics, no valid extractions for MAE"

    # Combine all failures for the failures field
    all_failures = critical_failures + important_failures + failures

    # Score is the fraction of checked metrics transcribed within tolerance.
    #
    # It deliberately is NOT derived from the mean absolute error: that would
    # average a glucose error in mg/dL with a CV error in percentage points and
    # a unitless risk index, which is not a meaningful quantity. A fraction is
    # comparable across metrics and cannot disagree with the pass/fail verdict.
    n_checked = len(checked_metrics)
    n_within_tolerance = n_checked - len(all_failures)
    score = (n_within_tolerance / n_checked) if n_checked else 0.0

    return EvalResult(
        pass_=passed,
        score=score,
        summary=summary,
        failures=all_failures,
        # Extra fields
        critical_failures=critical_failures,
        important_failures=important_failures,
        other_failures=failures,
        mae_details=mae_results,
        extracted_metrics=extracted_metrics,
        overall_mae=overall_mae
    )


if __name__ == "__main__":
    # Score a generated report against the bundled synthetic participant:
    #   python run_report.py
    #   python -m evals.evaluators.numerical out/report.md
    import sys
    from pathlib import Path

    from evals.core.utils import load_ground_truth, load_report

    repo_root = Path(__file__).resolve().parents[2]
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "out" / "report.md"
    gt_path = repo_root / "data" / "synthetic_001" / "ground_truth.json"

    if not report_path.exists():
        print(f"No report at {report_path}. Generate one with: python run_report.py")
        sys.exit(1)
    if not gt_path.exists():
        print(f"No ground truth at {gt_path}. Generate it with: "
              "python run_report.py --write-ground-truth")
        sys.exit(1)

    result = eval_numerical(load_report(str(report_path)), load_ground_truth(str(gt_path)))
    print(f"{'PASS' if result.pass_ else 'FAIL'}  score={result.score:.2f}  {result.summary}")
    for failure in result.failures:
        print(f"  - {failure}")
