"""Auto-discovery registry for @eval-decorated evaluation functions.

Eval Registry

Scans evals/evaluators/*.py files for functions decorated with @eval.
Provides filtering by category, requires_llm, requires_ground_truth.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class EvalInfo:
    """Metadata about a registered eval function."""
    name: str
    func: Callable
    version: str
    categories: List[str]
    requires_llm: bool
    requires_ground_truth: bool
    description: str


# Module-level registry (populated on first access)
_registry: Dict[str, EvalInfo] = {}
_discovered: bool = False


def _discover_evals() -> Dict[str, EvalInfo]:
    """Auto-discover all @eval-decorated functions in evals/ directory.

    Scans all .py files (excluding framework files) for functions with
    __eval_name__ attribute set by the @eval decorator.

    Returns:
        Dictionary mapping eval name to EvalInfo
    """
    evals_found = {}

    # Get the evaluators directory path (sibling to core/)
    evaluators_dir = Path(__file__).parent.parent / 'evaluators'

    # Scan all .py files in evaluators/
    for eval_file in evaluators_dir.glob('*.py'):
        if eval_file.name == '__init__.py':
            continue

        # Import the module from evals.evaluators package
        module_name = f"evals.evaluators.{eval_file.stem}"
        try:
            module = importlib.import_module(module_name)

            # Find all @eval-decorated functions in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # Check if it has the @eval decorator (has __eval_name__ attribute)
                if callable(attr) and hasattr(attr, '__eval_name__'):
                    eval_info = EvalInfo(
                        name=attr.__eval_name__,
                        func=attr,
                        version=getattr(attr, '__version__', '0.0.0'),
                        categories=getattr(attr, '__categories__', []),
                        requires_llm=getattr(attr, '__requires_llm__', False),
                        requires_ground_truth=getattr(attr, '__requires_ground_truth__', False),
                        description=getattr(attr, '__description__', '')
                    )
                    evals_found[eval_info.name] = eval_info

        except Exception as e:
            print(f"Warning: Failed to import eval from {eval_file.stem}: {e}")

    return evals_found


def _ensure_discovered() -> None:
    """Ensure evals have been discovered."""
    global _registry, _discovered
    if not _discovered:
        _registry = _discover_evals()
        _discovered = True


def get_eval(name: str) -> Optional[EvalInfo]:
    """Get eval by name.

    Args:
        name: Eval function name (e.g., 'eval_citation')

    Returns:
        EvalInfo or None if not found
    """
    _ensure_discovered()
    return _registry.get(name)


def list_evals() -> List[EvalInfo]:
    """List all registered evals.

    Returns:
        List of all EvalInfo objects
    """
    _ensure_discovered()
    return list(_registry.values())


def filter_evals(
    category: Optional[str] = None,
    requires_llm: Optional[bool] = None,
    requires_ground_truth: Optional[bool] = None
) -> List[EvalInfo]:
    """Filter evals by criteria.

    Args:
        category: Filter by category (e.g., 'citation', 'cgm')
        requires_llm: Filter by LLM requirement (True/False)
        requires_ground_truth: Filter by ground truth requirement (True/False)

    Returns:
        List of matching EvalInfo objects
    """
    _ensure_discovered()

    results = list(_registry.values())

    if category is not None:
        results = [e for e in results if category in e.categories]

    if requires_llm is not None:
        results = [e for e in results if e.requires_llm == requires_llm]

    if requires_ground_truth is not None:
        results = [e for e in results if e.requires_ground_truth == requires_ground_truth]

    return results


def reset_registry() -> None:
    """Reset the registry (for testing)."""
    global _registry, _discovered
    _registry = {}
    _discovered = False


if __name__ == "__main__":
    # Quick test
    print("Discovering evals...")
    evals = list_evals()

    print(f"\nFound {len(evals)} evals:\n")
    for e in sorted(evals, key=lambda x: x.name):
        print(f"  {e.name} v{e.version}")
        print(f"    Categories: {', '.join(e.categories)}")
        print(f"    Requires LLM: {e.requires_llm}")
        print(f"    Requires GT: {e.requires_ground_truth}")
        print(f"    Description: {e.description}")
        print()

    # Test filtering
    print("\n--- Non-LLM evals ---")
    for e in filter_evals(requires_llm=False):
        print(f"  {e.name}")

    print("\n--- CGM category evals ---")
    for e in filter_evals(category='cgm'):
        print(f"  {e.name}")
