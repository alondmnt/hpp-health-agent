"""Agent memory initialization and management.

This module provides functions for creating and managing the agent's memory state,
which tracks user prompts, context, tool outputs, and execution metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_initial_memory() -> Dict[str, Any]:
    """Build the initial memory structure for the agent.
    
    Returns:
        Dictionary containing empty/default values for all memory fields:
        - user_prompt: User's input request
        - context_docs: Retrieved RAG documents
        - tool_events: History of tool executions
        - report_outline: Planned structure for the report
        - final_report_path: Path to the rendered report
        - provenance: Execution metadata (seed, model, timings, etc.)
    """
    return {
        "user_prompt": None,
        "context_docs": [],
        "tool_events": [],
        "report_outline": None,
        "final_report_path": None,
        "provenance": {
            "seed": 0,
            "model_id": None,
            "temperature": 0.0,
            "tool_versions": {},
            "timings": {},
        },
    }


