#!/usr/bin/env python
"""CLI runner for the eval framework.

Commands:
    list    List available evals with metadata
    run     Run evals on a report
    info    Show detailed info about an eval

Usage:
    python runner.py list [--json] [--category CAT] [--no-llm]
    python runner.py run --report PATH [--ground-truth PATH] [--sample-id ID] [--system-id ID] [--category CAT] [--skip NAME] [--no-llm] [-o OUTPUT]
    python runner.py info NAME
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml

from evals.core.registry import list_evals, get_eval, filter_evals, EvalInfo
from evals.core.utils import load_report, load_ground_truth

def load_config(config_path: Optional[str]) -> Dict:
    """Load YAML config file. Returns empty dict if path is None or file doesn't exist."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def cmd_list(args: argparse.Namespace) -> int:
    """List available evals."""
    evals = filter_evals(
        category=args.category,
        requires_llm=None if not hasattr(args, 'no_llm') or not args.no_llm else False
    )

    if args.json:
        output = []
        for e in sorted(evals, key=lambda x: x.name):
            output.append({
                'name': e.name,
                'version': e.version,
                'categories': e.categories,
                'requires_llm': e.requires_llm,
                'requires_ground_truth': e.requires_ground_truth,
                'description': e.description
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"\nAvailable evals ({len(evals)}):\n")
        for e in sorted(evals, key=lambda x: x.name):
            llm_tag = " [LLM]" if e.requires_llm else ""
            gt_tag = " [GT]" if e.requires_ground_truth else ""
            print(f"  {e.name} v{e.version}{llm_tag}{gt_tag}")
            print(f"    {e.description}")
            print(f"    Categories: {', '.join(e.categories)}")
            print()

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info about an eval."""
    eval_info = get_eval(args.name)

    if eval_info is None:
        print(f"Error: Eval '{args.name}' not found", file=sys.stderr)
        return 1

    print(f"\n{eval_info.name}")
    print("=" * len(eval_info.name))
    print(f"Version: {eval_info.version}")
    print(f"Description: {eval_info.description}")
    print(f"Categories: {', '.join(eval_info.categories)}")
    print(f"Requires LLM: {eval_info.requires_llm}")
    print(f"Requires Ground Truth: {eval_info.requires_ground_truth}")
    print()

    return 0


def run_single_eval(
    eval_info: EvalInfo,
    report_text: str,
    ground_truth: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Dict:
    """Run a single eval and return result dict."""
    try:
        result = eval_info.func(report_text, ground_truth, config=config)
        output = result.model_dump(by_alias=True)
        output["categories"] = eval_info.categories
        return output
    except Exception as e:
        return {
            'pass': False,
            'score': 0.0,
            'summary': f"Error: {str(e)}",
            'failures': [str(e)],
            'categories': eval_info.categories
        }


def average_result_field(results: Dict[str, Dict[str, Any]], field_name: str) -> Optional[float]:
    """Average a numeric result field, ignoring None and bool values."""
    values = []
    for result in results.values():
        value = result.get(field_name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _config_list(value: Any) -> List[str]:
    """Normalize string/list config values into a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _system_id_contains(system_id: Optional[str], fragments: List[str]) -> bool:
    """Check whether any configured fragment appears in the system_id."""
    if not system_id:
        return False
    return any(fragment in system_id for fragment in fragments)


def eval_skip_reason(eval_config: Dict[str, Any], system_id: Optional[str]) -> Optional[str]:
    """Return config-driven skip reason for an eval/system pair."""
    skip_reason = eval_config.get("skip_reason", "not applicable for this condition")

    skip_fragments = _config_list(eval_config.get("skip_if_system_id_contains"))
    if _system_id_contains(system_id, skip_fragments):
        return skip_reason

    required_fragments = _config_list(eval_config.get("only_if_system_id_contains"))
    if required_fragments and not _system_id_contains(system_id, required_fragments):
        return skip_reason

    return None


def skipped_eval_result(eval_info: EvalInfo, reason: str) -> Dict[str, Any]:
    """Build a runner result for an intentionally skipped eval."""
    return {
        "pass": True,
        "score": None,
        "summary": f"Skipped - {reason}",
        "failures": [],
        "categories": eval_info.categories,
        "skipped": True,
        "skip_reason": reason,
    }


def cmd_run(args: argparse.Namespace) -> int:
    """Run evals on a report."""
    # Load report
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report not found: {args.report}", file=sys.stderr)
        return 1

    report_text = load_report(str(report_path))

    # Load config if provided
    config = load_config(getattr(args, 'config', None))

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            print(f"Error: Ground truth not found: {args.ground_truth}", file=sys.stderr)
            return 1
        ground_truth = load_ground_truth(str(gt_path))

    trace_path = None
    if trace_path:
        if ground_truth is None:
            ground_truth = {}
        ground_truth['trace_path'] = trace_path
        if not args.json:
            print(f"Trace discovered: {Path(trace_path).name}")

    # Get evals to run
    evals = filter_evals(
        category=args.category,
        requires_llm=False if args.no_llm else None,
        requires_ground_truth=None
    )

    # Apply skip filter
    skip_names = set(args.skip) if args.skip else set()
    evals = [e for e in evals if e.name not in skip_names]

    # Skip ground-truth-requiring evals if no GT provided
    if ground_truth is None:
        skipped_gt = [e.name for e in evals if e.requires_ground_truth]
        evals = [e for e in evals if not e.requires_ground_truth]
        if skipped_gt and not args.json:
            print(f"Skipping (no ground truth): {', '.join(skipped_gt)}\n")

    if not evals:
        print("No evals to run", file=sys.stderr)
        return 1

    # Run evals
    results = {}
    for eval_info in sorted(evals, key=lambda x: x.name):
        if not args.json:
            print(f"Running {eval_info.name}...", end=" ", flush=True)

        # Get config section for this eval (empty dict if not present)
        eval_config = dict(config.get(eval_info.name, {}))
        if args.system_id:
            eval_config["_system_id"] = args.system_id
        if args.sample_id:
            eval_config["_sample_id"] = args.sample_id
        skip_reason = eval_skip_reason(eval_config, args.system_id)
        if skip_reason is not None:
            result = skipped_eval_result(eval_info, skip_reason)
        else:
            result = run_single_eval(eval_info, report_text, ground_truth, config=eval_config)
        results[eval_info.name] = result

        if not args.json:
            status = "SKIP" if result.get("skipped") else ("PASS" if result['pass'] else "FAIL")
            score = result['score']
            score_str = f"{score:.2f}" if score is not None else "N/A"
            print(f"{status} (score: {score_str})")

    # Calculate overall (exclude None scores from average)
    all_pass = all(r['pass'] for r in results.values())
    valid_scores = [r['score'] for r in results.values() if r['score'] is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    content_score = average_result_field(results, "content_score")
    tool_execution_score = average_result_field(results, "tool_execution_score")

    # Output
    output_data = {
        'sample_id': args.sample_id,
        'system_id': args.system_id,
        'timestamp': datetime.now().isoformat(),
        'report_path': str(report_path),
        'ground_truth_path': str(args.ground_truth) if args.ground_truth else None,
        'overall_pass': all_pass,
        'overall_score': avg_score,
        'content_score': content_score,
        'tool_execution_score': tool_execution_score,
        'results': results
    }

    if args.json:
        print(json.dumps(output_data, indent=2))
    else:
        print()
        print("=" * 60)
        print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Passed: {sum(1 for r in results.values() if r['pass'])}/{len(results)}")

        # Show failures
        failed = [(name, r) for name, r in results.items() if not r['pass']]
        if failed:
            print("\nFailures:")
            for name, r in failed:
                print(f"\n  {name}:")
                print(f"    {r['summary']}")
                for f in r['failures'][:3]:
                    # Strip leading bullet if present to avoid double bullets
                    f_clean = f.lstrip('- ').lstrip()
                    print(f"    - {f_clean[:80]}...")

    # Write to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        if not args.json:
            print(f"\nResults written to: {args.output}")

    return 0 if all_pass else 1


def main():
    parser = argparse.ArgumentParser(
        description="Eval Framework CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # list command
    list_parser = subparsers.add_parser('list', help='List available evals')
    list_parser.add_argument('--json', action='store_true', help='Output as JSON')
    list_parser.add_argument('--category', type=str, help='Filter by category')
    list_parser.add_argument('--no-llm', action='store_true', help='Only non-LLM evals')

    # run command
    run_parser = subparsers.add_parser('run', help='Run evals on a report')
    run_parser.add_argument('--report', '-r', required=True, help='Path to report file')
    run_parser.add_argument('--ground-truth', '-g', help='Path to ground truth JSON')
    run_parser.add_argument('--config', help='Path to eval config YAML')
    run_parser.add_argument('--sample-id', help='Label for the sample, recorded in the output (e.g. synthetic_001)')
    run_parser.add_argument('--system-id', help='Label for the system under test, recorded in the output (e.g. agent, no-skills)')
    run_parser.add_argument('--category', '-c', help='Filter by category')
    run_parser.add_argument('--skip', '-s', action='append', help='Skip eval by name')
    run_parser.add_argument('--no-llm', action='store_true', help='Skip LLM evals')
    run_parser.add_argument('--output', '-o', help='Output JSON file')
    run_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # info command
    info_parser = subparsers.add_parser('info', help='Show eval details')
    info_parser.add_argument('name', help='Eval name')

    args = parser.parse_args()

    if args.command == 'list':
        return cmd_list(args)
    elif args.command == 'run':
        return cmd_run(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
