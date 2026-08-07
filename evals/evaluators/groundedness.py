"""LLM-based eval for groundedness of glucose event claims.

Uses DSPy to check whether event claims carry temporal references.

Evaluation: Groundedness (LLM)

Checks whether glucose event claims in the report are grounded with temporal references.
- Event claims (spikes, lows, per-day metrics) require dates or clear day references
- Summary claims (mean, overall TIR, CV) are skipped

"""

from typing import Dict, Optional

from evals.core.eval_decorator import eval, EvalResult

# Lazy imports for DSPy (only when needed)
_dspy = None
_get_lm = None

# Cached signature class (created once after DSPy import)
_GROUNDEDNESS_SIGNATURE = None


# =============================================================================
# Constants
# =============================================================================

GROUNDING_PASS_THRESHOLD = 0.95  # 95% of event claims must be grounded


# =============================================================================
# DSPy Setup
# =============================================================================

def _ensure_dspy_imports():
    """Lazy import DSPy dependencies."""
    global _dspy, _get_lm
    if _dspy is None:
        import dspy
        from pha.llm import get_lm
        _dspy = dspy
        _get_lm = get_lm


def _get_groundedness_signature():
    """Get or create DSPy signature class for groundedness assessment.

    Caches the signature class after first creation.
    Must be called after _ensure_dspy_imports() to ensure dspy is available.
    """
    global _GROUNDEDNESS_SIGNATURE
    if _GROUNDEDNESS_SIGNATURE is not None:
        return _GROUNDEDNESS_SIGNATURE

    class GroundednessJudgeSignature(_dspy.Signature):
        """Assess whether glucose event claims in a health report are grounded with dates.

        TASK: Find glucose event claims and check if they have temporal references.

        EVENT CLAIMS (check for dates):
        - Specific glucose readings/excursions: spikes, lows, peaks, drops with values
          Example: "spiked to 149 mg/dL", "dropped to 3.4 mmol/L"
        - Per-day or per-time-block metrics: TIR, time below range for specific days
          Example: "TIR on July 1st was lower", "time below range was high that day"

        SUMMARY CLAIMS (skip, don't count):
        - Global/aggregate stats over entire period: mean, median, CV, overall TIR
          Example: "Mean glucose was 100 mg/dL", "Overall TIR was 93%"
        - Typical/general patterns: "Fasting glucose is typically 88-96 mg/dL"

        GROUNDED means the event claim has a clear time anchor:
        - Explicit date: "June 22", "2020-06-22", "July 1st"
        - Day reference: "on that day", "on day 3", "on Thursday"
        - Well-defined period: "during the first weekend" (if clearly defined)

        UNGROUNDED means no clear time anchor:
        - "you sometimes spike after dinner" (vague)
        - "TIR was lower on some days" (no specifics)
        - "recently", "occasionally"

        IGNORE: Markdown tables, code fences, pure metric tables/boilerplate.
        Only judge narrative text.

        Treat both mg/dL and mmol/L as glucose units.

        Return JSON with counts. Do NOT include score/pass - framework computes those.
        """
        report_text: str = _dspy.InputField(desc="Full report markdown/text to evaluate")
        response_json: str = _dspy.OutputField(
            desc="JSON with keys: total_event_claims (int), grounded_event_claims (int), "
                 "ungrounded_claims (int), skipped_summary_claims (int), "
                 "ungrounded_examples (list of up to 5 claim snippets)"
        )

    _GROUNDEDNESS_SIGNATURE = GroundednessJudgeSignature
    return _GROUNDEDNESS_SIGNATURE


def _parse_json_response(text: str, fallback: Dict) -> Dict:
    """Robust JSON parser that tolerates extra text around JSON payloads."""
    import json
    import re

    if not text:
        return fallback

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return fallback
        return fallback


# =============================================================================
# Main Eval
# =============================================================================

@eval(
    version="1.0.0",
    categories=["groundedness", "quality"],
    requires_ground_truth=False,
    requires_llm=True,
    description="Check glucose event claims have date references (LLM judge)"
)
def eval_groundedness_llm(
    report_text: str,
    ground_truth: Optional[Dict] = None,
    config: Optional[Dict] = None,
    model: Optional[str] = None
) -> EvalResult:
    """Evaluate groundedness of glucose event claims using LLM-as-judge.

    Uses DSPy to assess whether glucose event claims (spikes, lows, per-day
    metrics) have temporal references (dates, day references, defined periods).
    Summary claims (mean, overall TIR, CV) are skipped.

    Args:
        report_text: Full report text
        ground_truth: Not used for this eval; included only for signature
            consistency with other evals.
        config: Optional config dict with 'pass_threshold' key
        model: Optional model override (e.g., 'anthropic/claude-sonnet-4-20250514')

    Returns:
        EvalResult with:
        - pass_: True if >= 95% of event claims are grounded
        - score: Fraction of event claims grounded (0-1)
        - summary: "{grounded}/{total} event claims grounded"
        - failures: List of up to 5 ungrounded claim examples
    """
    cfg = config or {}
    pass_threshold = cfg.get('pass_threshold', GROUNDING_PASS_THRESHOLD)

    _ensure_dspy_imports()

    # Get cached signature class
    GroundednessJudgeSignature = _get_groundedness_signature()

    # Create predictor with custom LM
    # MID tier for LLM-as-judge - evaluation accuracy matters
    predictor = _dspy.Predict(GroundednessJudgeSignature)
    predictor.lm = _get_lm(tier="MID", model_override=model)

    # Call predictor with report text
    sample = predictor(report_text=report_text)

    default_payload = {
        "total_event_claims": 0,
        "grounded_event_claims": 0,
        "ungrounded_claims": 0,
        "skipped_summary_claims": 0,
        "ungrounded_examples": [],
    }
    result = _parse_json_response(sample.response_json, default_payload)

    # Extract counts from LLM response with sanity clamping
    total_event_claims = max(0, int(result.get("total_event_claims", 0)))
    grounded_event_claims = max(0, int(result.get("grounded_event_claims", 0)))
    ungrounded_claims = max(0, int(result.get("ungrounded_claims", 0)))
    skipped_summary_claims = max(0, int(result.get("skipped_summary_claims", 0)))
    ungrounded_examples = result.get("ungrounded_examples", [])[:5]

    # Sanity: grounded cannot exceed total
    if grounded_event_claims > total_event_claims:
        grounded_event_claims = total_event_claims

    # Framework computes score/pass (don't trust LLM's values)
    if total_event_claims == 0:
        score = 1.0
    else:
        score = grounded_event_claims / total_event_claims

    passed = score >= pass_threshold

    summary = f"{grounded_event_claims}/{total_event_claims} event claims grounded"

    # Format failures with snippets
    failures = [f"Ungrounded: {ex}" for ex in ungrounded_examples]

    # Build extra fields dict for Pydantic's extra="allow"
    extra_fields = {
        "total_event_claims": total_event_claims,
        "grounded_event_claims": grounded_event_claims,
        "ungrounded_claims": ungrounded_claims,
        "skipped_summary_claims": skipped_summary_claims,
        "ungrounded_examples": ungrounded_examples,
        "method": "llm_judge",
        "model": model or "default",
    }

    return EvalResult(
        pass_=passed,
        score=score,
        summary=summary,
        failures=failures,
        **extra_fields,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Judge a generated report:
    #   python run_report.py
    #   python -m evals.evaluators.groundedness out/report.md
    import sys
    from pathlib import Path

    from evals.core.utils import load_report

    repo_root = Path(__file__).resolve().parents[2]
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "out" / "report.md"

    if not report_path.exists():
        print(f"No report at {report_path}. Generate one with: python run_report.py")
        sys.exit(1)

    result = eval_groundedness_llm(load_report(str(report_path)))
    print(f"{'PASS' if result.pass_ else 'FAIL'}  score={result.score:.2f}  {result.summary}")
