"""Utilities for saving and analyzing execution traces.

This module provides functions to:
- Save comprehensive traces including LLM interactions
- Load and analyze trace files
- Extract costs, timing, and token usage
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional



def save_trace(
    memory: Dict[str, Any],
    lm: Optional[Any] = None,
    trace_dir: Optional[Path] = None,
) -> Path:
    """
    Save comprehensive execution trace including:
    - Memory state (tool calls, provenance)
    - LLM history (all prompts, responses, reasoning)
    - Timestamps and costs

    Args:
        memory: The agent's memory dictionary
        lm: The DSPy LM instance
        trace_dir: Directory to save traces (defaults to ./traces)

    Returns:
        Path to saved trace file
    """
    if trace_dir is None:
        trace_dir = Path("traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = trace_dir / f"trace_{timestamp}.json"
    
    # Build comprehensive trace
    trace = {
        "metadata": {
            "timestamp": timestamp,
            "trace_version": "1.1",
            "system": "pha",
        },

        # User request and context
        "user_prompt": memory.get("user_prompt"),
        "context_docs": memory.get("context_docs", []),

        # Input data paths
        "data_paths": memory.get("data_paths"),
        "csv_path": memory.get("csv_path"),

        # Execution plan and tool calls
        "plan": memory.get("plan"),
        "tool_events": memory.get("tool_events", []),

        # Artifact mapping
        "artifact_map": memory.get("artifact_map", {}),

        # Analysis output
        "analysis_text": memory.get("analysis_text"),

        # Report info
        "report_outline": memory.get("report_outline"),
        "final_report_path": memory.get("final_report_path"),

        # Provenance and timing
        "provenance": memory.get("provenance", {}),

        # LLM interaction history (THE KEY PART!)
        "llm_history": [],
    }
    
    # Add LLM history with all prompts and responses
    if lm is not None and hasattr(lm, 'history'):
        for entry in lm.history:
            llm_entry = {
                "timestamp": entry.get("timestamp"),
                "uuid": entry.get("uuid"),
                "model": entry.get("model"),
                "response_model": entry.get("response_model"),
                
                # The actual prompts sent
                "messages": entry.get("messages", []),
                "prompt": entry.get("prompt"),
                
                # The model's responses
                "outputs": entry.get("outputs", []),
                
                # Token usage and cost
                "usage": entry.get("usage", {}),
                "cost": entry.get("cost"),
                
                # Parameters used
                "kwargs": entry.get("kwargs", {}),
            }
            trace["llm_history"].append(llm_entry)
    
    
    # Save to file
    with open(trace_file, 'w') as f:
        json.dump(trace, f, indent=2, default=str)
    
    print(f"✓ Trace saved to: {trace_file}")
    print(f"  - LLM calls captured: {len(trace['llm_history'])}")
    print(f"  - Tool events: {len(trace['tool_events'])}")
    print(f"  - Total size: {trace_file.stat().st_size / 1024:.1f} KB")
    
    return trace_file


def load_trace(trace_file: Path | str) -> Dict[str, Any]:
    """Load a saved trace file.
    
    Args:
        trace_file: Path to the trace JSON file
        
    Returns:
        Dictionary containing the trace data
    """
    with open(trace_file, 'r') as f:
        return json.load(f)


def analyze_trace_costs(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate total cost and token usage from a trace.
    
    Args:
        trace: Loaded trace dictionary
        
    Returns:
        Dictionary with cost and token statistics
    """
    total_cost = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for llm_call in trace.get('llm_history', []):
        if llm_call.get('cost'):
            total_cost += llm_call['cost']
        
        usage = llm_call.get('usage', {})
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        total_tokens += usage.get('total_tokens', 0)
    
    return {
        'total_cost': total_cost,
        'total_tokens': total_tokens,
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'llm_calls': len(trace.get('llm_history', [])),
    }


def extract_all_prompts(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all prompts sent to LLMs.
    
    Args:
        trace: Loaded trace dictionary
        
    Returns:
        List of prompt dictionaries with metadata
    """
    prompts = []
    for i, llm_call in enumerate(trace.get('llm_history', []), 1):
        messages = llm_call.get('messages', [])
        prompts.append({
            'call_number': i,
            'timestamp': llm_call.get('timestamp'),
            'model': llm_call.get('model'),
            'messages': messages,
        })
    return prompts


def extract_all_responses(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all LLM responses.
    
    Args:
        trace: Loaded trace dictionary
        
    Returns:
        List of response dictionaries with metadata
    """
    responses = []
    for i, llm_call in enumerate(trace.get('llm_history', []), 1):
        responses.append({
            'call_number': i,
            'timestamp': llm_call.get('timestamp'),
            'model': llm_call.get('model'),
            'output': llm_call.get('outputs', []),
        })
    return responses


def print_trace_summary(trace: Dict[str, Any]) -> None:
    """Print a formatted summary of a trace.
    
    Args:
        trace: Loaded trace dictionary
    """
    print("=" * 80)
    print("TRACE SUMMARY")
    print("=" * 80)
    
    # Metadata
    metadata = trace.get('metadata', {})
    print(f"\nTimestamp: {metadata.get('timestamp')}")
    print(f"System: {metadata.get('system')}")
    
    # User prompt
    prompt = trace.get('user_prompt', '')
    print(f"\nUser Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"\nUser Prompt: {prompt}")
    
    # Context
    context_docs = trace.get('context_docs', [])
    print(f"\nRAG Context: {len(context_docs)} documents retrieved")
    
    # Tools
    tool_events = trace.get('tool_events', [])
    print(f"\nTool Executions: {len(tool_events)}")
    for event in tool_events:
        print(f"  - {event.get('tool_name')}")
    
    # LLM interactions
    llm_history = trace.get('llm_history', [])
    print(f"\nLLM Interactions: {len(llm_history)} calls")
    
    # Cost analysis
    costs = analyze_trace_costs(trace)
    print(f"\n💰 Cost Analysis:")
    print(f"  Total cost: ${costs['total_cost']:.6f}")
    print(f"  Total tokens: {costs['total_tokens']:,}")
    print(f"  Prompt tokens: {costs['prompt_tokens']:,}")
    print(f"  Completion tokens: {costs['completion_tokens']:,}")
    if costs['llm_calls'] > 0:
        avg_tokens = costs['total_tokens'] / costs['llm_calls']
        print(f"  Avg tokens/call: {avg_tokens:.1f}")
    
    # Timing
    provenance = trace.get('provenance', {})
    timings = provenance.get('timings', {})
    if timings:
        print(f"\n⏱️  Timing Breakdown:")
        for component, ms in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {component:.<35} {ms:>8.0f} ms")
    
    # Output
    report_path = trace.get('final_report_path')
    if report_path:
        print(f"\n📄 Report: {report_path}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        trace_path = Path(sys.argv[1])
        if trace_path.exists():
            trace = load_trace(trace_path)
            print_trace_summary(trace)
        else:
            print(f"Trace file not found: {trace_path}")
    else:
        print("Usage: python trace_utils.py <path_to_trace.json>")

