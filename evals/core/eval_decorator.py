"""Decorator standardising eval function metadata and output.

Simpler than the tool decorator: metadata attachment and validation only.

Decorator that automatically:
1. Attaches metadata for discoverability
2. Validates return type is EvalResult
3. Preserves original function signature
4. Provides DSPy integration via .as_dspy_metric()

Usage:
    @eval(version="1.0.0", categories=["cgm", "completeness"], requires_llm=False)
    def eval_critical_cgm_metrics(report_text: str, ground_truth: Optional[Dict] = None) -> EvalResult:
        '''Check critical CGM metrics present in report.'''
        # ... evaluation logic ...
        return EvalResult(
            pass_=True,
            score=0.95,
            summary="3/3 metrics present",
            failures=[]
        )

    # Normal usage
    result = eval_critical_cgm_metrics(report_text, ground_truth)

    # DSPy optimizer usage
    metric_fn = eval_critical_cgm_metrics.as_dspy_metric()
    score_feedback = metric_fn(example, pred, trace, pred_name, pred_trace)

Design principles:
- Minimal - no wrappers, just metadata
- Standardized output schema
- Clear requirements (ground_truth? LLM?)
- Easy to discover and filter
- DSPy-ready for optimization
"""
from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps

from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T", bound=Callable)

# ============================================================================
# Eval Output Model
# ============================================================================

class EvalResult(BaseModel):
    """Standardized eval output.

    All eval functions must return this model for consistency.

    Attributes:
        pass_: Whether the evaluation passed (True) or failed (False)
                (aliased as "pass" in serialization to match existing eval output format)
        score: Numeric score, typically 0.0-1.0 (e.g., proportion of checks passed).
               None indicates eval was skipped (excluded from aggregation).
        summary: Human-readable summary of results
        failures: List of specific failure messages (empty if passed)

    Extra fields are allowed for eval-specific details (e.g., extracted_metrics,
    uncited_claims, etc.). These can be accessed but won't fail validation.

    Note: Use result.pass_ to access pass/fail status in code.
          Use result.model_dump(by_alias=True) to get {"pass": True, ...} for serialization.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    pass_: bool = Field(..., alias="pass", description="Whether evaluation passed")
    score: Optional[float] = Field(..., description="Numeric score (typically 0.0-1.0), None if skipped")
    summary: str = Field(..., description="Human-readable summary")
    failures: List[str] = Field(default_factory=list, description="List of failure messages")


# ============================================================================
# Decorator Implementation
# ============================================================================

__all__ = [
    "eval",
    "EvalResult",
]


# ============================================================================
# DSPy Integration
# ============================================================================

def _create_dspy_metric_wrapper(eval_fn: Callable) -> Callable:
    """Create DSPy-compatible metric function from eval function.

    Returns a metric function with signature:
        metric(example, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback

    The wrapper:
    1. Extracts report_text and ground_truth from example/pred
    2. Calls the original eval function
    3. Stores full EvalResult in eval_fn._per_example
    4. Converts EvalResult to ScoreWithFeedback for GEPA

    Args:
        eval_fn: Decorated eval function that returns EvalResult

    Returns:
        DSPy-compatible metric function
    """
    # Import ScoreWithFeedback once at wrapper creation time
    try:
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
        score_class = ScoreWithFeedback
    except ImportError:
        # Fallback if DSPy not installed or GEPA not available
        score_class = None

    def dspy_metric(**kwargs):
        """DSPy-compatible metric wrapper.

        Accepts keyword arguments only for GEPA compatibility:
        - example= or gold=
        - pred= or prediction=
        - trace=, pred_name=, pred_trace= (optional)
        """
        # Extract example and pred flexibly (GEPA may use different names)
        example = kwargs.get("example") or kwargs.get("gold")
        pred = kwargs.get("pred") or kwargs.get("prediction")

        if example is None or pred is None:
            raise ValueError(
                f"DSPy metric for {eval_fn.__name__} requires 'example' and 'pred' keyword arguments. "
                f"Got kwargs={list(kwargs.keys())}"
            )

        # Extract report_text (try pred first, fallback to example)
        # For report generation: pred.report_text (model output)
        # For report evaluation: example.report_text (input)
        report_text = getattr(pred, "report_text", None)
        if report_text is None:
            report_text = getattr(example, "report_text", None)

        if report_text is None:
            raise ValueError(
                f"DSPy metric for {eval_fn.__name__} requires 'report_text' field "
                f"in either pred or example. Available pred fields: {dir(pred)}, "
                f"available example fields: {dir(example)}"
            )

        # Extract ground_truth from example
        ground_truth = getattr(example, "ground_truth", None)

        # Validate ground_truth if required
        if eval_fn.__requires_ground_truth__ and ground_truth is None:
            raise ValueError(
                f"DSPy metric for {eval_fn.__name__} requires ground_truth "
                f"(marked as requires_ground_truth=True), but got None"
            )

        # Call original eval function with error handling
        try:
            result = eval_fn(report_text, ground_truth)
        except Exception as e:
            raise RuntimeError(
                f"DSPy metric for {eval_fn.__name__} failed during evaluation: {e}"
            ) from e

        # Store full result for inspection
        eval_fn._per_example.append({
            "example": example,
            "pred": pred,
            "result": result,
            "trace": kwargs.get("trace"),
            "pred_name": kwargs.get("pred_name"),
            "pred_trace": kwargs.get("pred_trace"),
        })

        # Convert EvalResult to rich feedback text
        feedback = result.summary
        if result.failures:
            feedback += "\n\nFailures:\n" + "\n".join(f"• {f}" for f in result.failures)

        # Return ScoreWithFeedback or dict fallback
        if score_class is not None:
            return score_class(score=result.score, feedback=feedback)
        else:
            # Return dict with score/feedback (compatible with most DSPy versions)
            return {"score": result.score, "feedback": feedback}

    return dspy_metric


def _pick_description(func: Callable, explicit: Optional[str]) -> str:
    """Pick eval description with fallback precedence.

    Precedence:
    1. Explicit description from @eval(description="...")
    2. First line of function docstring
    3. Humanized function name

    Args:
        func: Function to get description for
        explicit: Explicit description from decorator arg

    Returns:
        Eval description string
    """
    if explicit and explicit.strip():
        return explicit.strip()

    if func.__doc__:
        # Get first line (until first newline)
        first_line = func.__doc__.strip().split("\n", 1)[0].strip()
        if first_line:
            return first_line

    # Fallback: humanize function name
    return func.__name__.replace("_", " ").capitalize()


def eval(
    *,
    version: str = "1.0.0",
    categories: Union[List[str], str] = "Health",
    requires_ground_truth: bool = False,
    requires_llm: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[T], T]:
    """Decorator to attach metadata to eval functions.

    This decorator:
    1. Attaches metadata as dunder attributes for discoverability
    2. Validates return type is EvalResult
    3. Adds .as_dspy_metric() method for DSPy optimization integration
    4. Preserves original function (no wrappers)

    Args:
        version: Eval version (semantic versioning)
        categories: Eval categories (string or list)
        requires_ground_truth: Whether eval needs ground_truth parameter
        requires_llm: Whether eval makes LLM calls (for cost tracking)
        name: Override eval name (defaults to function name)
        description: Override description (uses fallback precedence)

    Returns:
        Decorated function (unchanged, with metadata attached)

    Example:
        @eval(
            version="1.0.0",
            categories=["cgm", "completeness"],
            requires_ground_truth=False,
            requires_llm=False
        )
        def eval_critical_cgm_metrics(
            report_text: str,
            ground_truth: Optional[Dict] = None
        ) -> EvalResult:
            '''Check that critical CGM metrics are present.'''
            # ... evaluation logic ...
            return EvalResult(pass_=True, score=1.0, summary="All present", failures=[])
    """
    def decorator(func: T) -> T:
        # Note: Type signature claims to return T but actually adds attributes
        # (.as_dspy_metric, ._per_example, __eval_name__, etc.)
        # Type checkers won't know about these - this is a limitation of Python decorators
        # Normalize categories to list
        if isinstance(categories, str):
            _categories = [categories]
        elif isinstance(categories, list):
            _categories = categories
        else:
            raise TypeError(f"categories must be str or list, got {type(categories)}")

        # Get eval name
        eval_name = name or func.__name__

        # Get description with fallback precedence
        eval_description = _pick_description(func, description)

        # Validate return type annotation
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        if return_annotation == inspect.Signature.empty:
            raise TypeError(f"{func.__name__} must have return type annotation (EvalResult)")

        # Check if return type is EvalResult (handle both direct type and string annotation)
        is_eval_result = False
        if return_annotation == EvalResult:
            is_eval_result = True
        elif isinstance(return_annotation, str) and return_annotation == "EvalResult":
            is_eval_result = True

        if not is_eval_result:
            raise TypeError(
                f"{func.__name__} must return EvalResult, "
                f"got {return_annotation}"
            )

        # Validate function signature has required parameters
        params = sig.parameters
        if "report_text" not in params:
            raise TypeError(
                f"{func.__name__} must have 'report_text' parameter"
            )

        if requires_ground_truth and "ground_truth" not in params:
            raise TypeError(
                f"{func.__name__} is marked requires_ground_truth=True "
                f"but has no 'ground_truth' parameter"
            )

        # Attach metadata to function (public API via dunders)
        setattr(func, "__eval_name__", eval_name)
        setattr(func, "__version__", version)
        setattr(func, "__categories__", _categories)
        setattr(func, "__requires_ground_truth__", requires_ground_truth)
        setattr(func, "__requires_llm__", requires_llm)
        setattr(func, "__description__", eval_description)

        # Add DSPy integration method
        def as_dspy_metric():
            """Return DSPy-compatible metric function for optimization.

            Returns a metric function that:
            - Accepts (example, pred, trace, pred_name, pred_trace)
            - Extracts report_text and ground_truth from example/pred
            - Calls the eval function
            - Stores full EvalResult in func._per_example
            - Returns ScoreWithFeedback for GEPA

            Usage:
                metric_fn = eval_fn.as_dspy_metric()
                score_feedback = metric_fn(example, pred, trace=None)
            """
            return _create_dspy_metric_wrapper(func)

        setattr(func, "as_dspy_metric", as_dspy_metric)

        # Initialize storage for per-example results (used by DSPy wrapper)
        setattr(func, "_per_example", [])

        # Return original function unchanged (no wrappers)
        return func

    return decorator
