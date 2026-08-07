"""Evaluation: critical CGM metrics included.


Verify report includes mandatory CGM metrics with values.
This is a basic completeness check - the report must contain:
- cgm_mean (mean glucose)
- cgm_cv (coefficient of variation)
- cgm_in_range_70_180 (time in range 70-180 mg/dL)

Pass criteria: All 3 critical metrics must be present with numeric values.
"""

from typing import Dict, Optional
from evals.core.eval_decorator import eval, EvalResult
from evals.core.utils import extract_metric_patterns


# Keys use cgm_* standard names (matching ground_truth.json)
MANDATORY_METRICS = ['cgm_mean', 'cgm_cv', 'cgm_in_range_70_180']


@eval(
    version="1.0.0",
    categories=["cgm", "completeness"],
    requires_ground_truth=False,
    requires_llm=False,
    description="Check critical CGM metrics (cgm_mean, cgm_cv, cgm_in_range_70_180) present"
)
def eval_critical_cgm_metrics(
    report_text: str,
    ground_truth: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> EvalResult:
    """
    Evaluate critical CGM metrics completeness.

    Checks that all mandatory metrics are present with numeric values.

    Args:
        report_text: Full report text
        ground_truth: Not used (included for signature consistency)
        config: Optional config dict with 'mandatory_metrics' list

    Returns:
        EvalResult with pass/fail and details of missing metrics
    """
    cfg = config or {}
    mandatory_metrics = cfg.get('mandatory_metrics', MANDATORY_METRICS)

    # Extract metrics from report
    extracted_metrics = extract_metric_patterns(report_text)

    # Check which mandatory metrics are present
    missing_metrics = []
    present_metrics = []

    for metric in mandatory_metrics:
        if extracted_metrics.get(metric) is None:
            missing_metrics.append(metric)
        else:
            present_metrics.append(metric)

    # Pass if all mandatory metrics present
    passed = len(missing_metrics) == 0

    # Calculate completeness score
    score = len(present_metrics) / len(mandatory_metrics)

    # Generate summary
    summary = f"{len(present_metrics)}/{len(mandatory_metrics)} critical metrics present"

    # Build failures list
    failures = [f"{m} not found in report" for m in missing_metrics]

    return EvalResult(
        pass_=passed,
        score=score,
        summary=summary,
        failures=failures,
        # Extra fields (allowed by model_config)
        missing_metrics=missing_metrics,
        present_metrics=present_metrics,
        extracted_values={m: extracted_metrics.get(m) for m in mandatory_metrics}
    )


if __name__ == "__main__":
    # Quick test to verify eval works
    sample_report = """
    CGM Analysis Report
    Mean glucose: 120.5 mg/dL
    CV: 35.2%
    Time in range 70-180: 75.0%
    """

    result = eval_critical_cgm_metrics(sample_report)

    print("=" * 60)
    print("CRITICAL CGM METRICS EVAL TEST")
    print("=" * 60)
    print(f"Status: {'✅ PASS' if result.pass_ else '❌ FAIL'}")
    print(f"Score: {result.score:.1%}")
    print(f"Summary: {result.summary}")
    if result.failures:
        print("\nFailures:")
        for f in result.failures:
            print(f"  - {f}")
    print("=" * 60)
