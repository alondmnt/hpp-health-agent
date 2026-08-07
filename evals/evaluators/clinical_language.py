"""Evaluation: clinical language compliance.

Verifies compliance with the clinical-language-guard skill by checking:
1. No forbidden terms (diabetic, dangerous, concerning, etc.)
2. No diagnostic phrases (pre-diabetic patterns, hypoglycemic episodes)
3. No urgent action language (see a doctor immediately, critical)

Based on rules defined in: skills/clinical_language_guard.md
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.core.eval_decorator import eval, EvalResult


# =============================================================================
# Rules from clinical_language_guard.md
# =============================================================================

# Forbidden terms - direct matches (case-insensitive)
FORBIDDEN_TERMS = [
    # Diagnostic labels (when used to describe the user)
    "diabetic",
    "pre-diabetic",
    "prediabetic",
    "hypoglycemic",
    "hyperglycemic",
    # Alarmist adjectives
    "dangerous",
    "alarming",
    "concerning",
    "worrying",
    "abnormal",
    "pathological",
    "diseased",
    # Severity markers (all caps or emphasized)
    "critical",
    "severe",
    "immediate attention",
]

# Forbidden phrases - pattern matches
FORBIDDEN_PATTERNS = [
    r"you\s+should\s+see\s+a\s+doctor\s+immediately",
    r"this\s+suggests?\s+\w+\s+disease",
    r"this\s+suggests?\s+diabetes",
    r"pre-?diabetic\s+pattern",
    r"hypoglycemic\s+episode",
    r"dangerous(ly)?\s+(high|low)",
    r"alarming(ly)?\s+(high|low)",
]


def _strip_references_section(text: str) -> str:
    """Remove references section from text to avoid false positives on citation titles.

    References sections typically start with "## References" or "### References"
    and contain academic citation titles that may include clinical terms.
    """
    # Find references section and remove everything after it
    patterns = [
        r'\n##\s*References\b.*',
        r'\n###\s*References\b.*',
        r'\n\*\*References:?\*\*.*',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _is_negated_context(text: str, match_start: int, term: str) -> bool:
    """Check if the term appears in a negated context.

    Handles patterns like:
    - "No severe hypoglycemia"
    - "No single day showed concerning patterns"
    - "reducing exposure to ... hypoglycemic episodes"
    - "without hypoglycemic episodes"
    """
    # Get context before the match (up to 60 chars)
    context_start = max(0, match_start - 60)
    context = text[context_start:match_start].lower()

    # Negation patterns that indicate the term is NOT describing the user
    negation_patterns = [
        r'\bno\s+(?:\w+\s+)*$',           # "No severe" or "No single day showed"
        r'\bwithout\s+(?:\w+\s+)*$',      # "without hypoglycemic"
        r'\breducing\s+(?:\w+\s+)*$',     # "reducing exposure to"
        r'\bexposure\s+to\s+(?:\w+\s+)*$', # "exposure to ... episodes"
        r'\bzero\s+(?:\w+\s+)*$',         # "zero severe"
        r'\bminimal\s+(?:\w+\s+)*$',      # "minimal hypoglycemia"
        r'\brare\s+(?:\w+\s+)*$',         # "rare hypoglycemic"
        r'\bavoid(?:ing|s|ed)?\s+(?:\w+\s+)*$',  # "avoiding hypoglycemic"
    ]

    for pattern in negation_patterns:
        if re.search(pattern, context):
            return True

    return False


def _check_forbidden_terms(text: str) -> List[Tuple[str, int]]:
    """Check for forbidden terms in text.

    Returns list of (term, count) tuples for terms found.
    Uses negative lookbehind to allow negated forms (e.g., "non-diabetic" is OK).
    Also excludes terms in negated contexts (e.g., "No severe hypoglycemia").
    """
    # Strip references section to avoid false positives on citation titles
    text_stripped = _strip_references_section(text)
    text_lower = text_stripped.lower()
    found = []

    for term in FORBIDDEN_TERMS:
        # Use word boundary matching with negative lookbehind for "non-"
        # This allows: "non-diabetic", "non-hypoglycemic", etc.
        pattern = r'(?<!non-)(?<!non)\b' + re.escape(term.lower()) + r'\b'

        # Find all matches and filter out negated contexts
        valid_matches = 0
        for match in re.finditer(pattern, text_lower):
            if not _is_negated_context(text_stripped, match.start(), term):
                valid_matches += 1

        if valid_matches > 0:
            found.append((term, valid_matches))

    return found


def _check_forbidden_patterns(text: str) -> List[Tuple[str, str]]:
    """Check for forbidden phrase patterns in text.

    Returns list of (pattern_desc, matched_text) tuples.
    Excludes references section and negated contexts.
    """
    # Strip references section
    text_stripped = _strip_references_section(text)
    text_lower = text_stripped.lower()
    found = []

    for pattern in FORBIDDEN_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            # Check if this is in a negated context
            if not _is_negated_context(text_stripped, match.start(), match.group(0)):
                found.append((pattern, match.group(0)))

    return found


@eval(
    version="1.0.0",
    categories=["skill", "clinical-language-guard", "language"],
    requires_ground_truth=False,
    requires_llm=False,
    description="Check compliance with clinical-language-guard skill",
)
def eval_clinical_language(
    report_text: str,
    ground_truth: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> EvalResult:
    """Check that report follows clinical-language-guard skill guidelines.

    Verifies:
    1. No forbidden terms (diabetic, dangerous, concerning, etc.)
    2. No diagnostic phrases (pre-diabetic patterns, suggests diabetes)
    3. No urgent action language (see doctor immediately)

    Args:
        report_text: Full report text to evaluate
        ground_truth: Not used for this eval
        config: Optional config overrides

    Returns:
        EvalResult with pass/fail, score, and list of violations
    """
    cfg = config or {}

    # Check for violations
    term_violations = _check_forbidden_terms(report_text)
    pattern_violations = _check_forbidden_patterns(report_text)

    # Build failure messages
    failures = []

    for term, count in term_violations:
        failures.append(f"Forbidden term '{term}' found {count}x")

    for pattern, matched in pattern_violations:
        failures.append(f"Forbidden phrase found: '{matched}'")

    # Calculate score
    total_checks = len(FORBIDDEN_TERMS) + len(FORBIDDEN_PATTERNS)
    violations_count = len(term_violations) + len(pattern_violations)

    # Score: 1.0 = perfect, decreases with violations
    # Each violation reduces score proportionally
    score = max(0.0, 1.0 - (violations_count / total_checks))

    # Summary
    if violations_count == 0:
        summary = "No clinical language violations found"
    else:
        summary = f"{violations_count} clinical language violation(s) found"

    return EvalResult(
        pass_=violations_count == 0,
        score=score,
        summary=summary,
        failures=failures,
        # Extra fields for debugging
        term_violations=term_violations,
        pattern_violations=pattern_violations,
    )


# =============================================================================
# CLI for standalone testing
# =============================================================================

if __name__ == "__main__":
    # Test with sample reports

    # BAD report (from skill docs)
    bad_report = """
    Your CGM data shows concerning patterns consistent with pre-diabetes. Your glucose
    variability is abnormally high and you experienced dangerous hypoglycemic episodes.
    You should consult an endocrinologist immediately.
    """

    # GOOD report (from skill docs)
    good_report = """
    Your 14-day CGM summary shows:
    - Average glucose: 142 mg/dL (typical non-diabetic range: 70-100 mg/dL)
    - Time in range (70-180 mg/dL): 68%, against a consensus target of above 70%
    - Glucose below 70 mg/dL: 3 episodes on Jan 5, 8, and 12

    These metrics may be worth discussing with your healthcare provider during
    your next visit.
    """

    print("=" * 70)
    print("CLINICAL LANGUAGE GUARD EVAL TEST")
    print("=" * 70)

    print("\n--- BAD REPORT (should FAIL) ---")
    result_bad = eval_clinical_language(bad_report)
    print(f"Status: {'PASS' if result_bad.pass_ else 'FAIL'}")
    print(f"Score: {result_bad.score:.1%}")
    print(f"Summary: {result_bad.summary}")
    if result_bad.failures:
        print("Failures:")
        for f in result_bad.failures:
            print(f"  - {f}")

    print("\n--- GOOD REPORT (should PASS) ---")
    result_good = eval_clinical_language(good_report)
    print(f"Status: {'PASS' if result_good.pass_ else 'FAIL'}")
    print(f"Score: {result_good.score:.1%}")
    print(f"Summary: {result_good.summary}")
    if result_good.failures:
        print("Failures:")
        for f in result_good.failures:
            print(f"  - {f}")

    print("\n" + "=" * 70)
    print("Test complete!")
