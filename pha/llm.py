"""LLM configuration.

Tiered LLM system with HIGH/MID/LOW capability levels. Models are resolved
through DSPy, which uses LiteLLM underneath, so any LiteLLM-supported provider
works (Anthropic, OpenAI, Bedrock, Vertex, OpenRouter, a local server, ...).

Default: Anthropic via the ANTHROPIC_API_KEY environment variable.

Config file location: <repo_root>/llm.json (optional; see llm.json.example)

Tiers:
  - HIGH: Complex reasoning, report generation  (max_tokens 32768)
  - MID:  Standard tasks, ReAct loops           (max_tokens 16384)
  - LOW:  Simple, high-volume, cost-sensitive   (max_tokens 8192)

  Override per call: get_lm(tier="LOW", max_tokens=500)

Default: Claude Opus 4.6, the foundation model used in the paper.

Example llm.json:

    {
      "tiers": {
        "HIGH": "anthropic/claude-opus-4-6",
        "MID":  "anthropic/claude-opus-4-6",
        "LOW":  "anthropic/claude-haiku-4-5"
      }
    }

To use a different provider, set the tier strings to that provider's LiteLLM
model IDs and export that provider's API key, for example:

    {"tiers": {"HIGH": "openai/gpt-5.2", "MID": "openai/gpt-5.2", "LOW": "openai/gpt-5.2-mini"}}

Any other key in llm.json is passed through to dspy.LM as a keyword argument,
which is how you point the agent at a self-hosted or proxied endpoint (api_base).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import dspy

ROOT_DIR = Path(__file__).resolve().parents[1]  # pha/ -> repo root
LLM_CONFIG_PATH = ROOT_DIR / "llm.json"

# Tier type for LLM capability levels
Tier = Literal["HIGH", "MID", "LOW"]

# Default models per tier (used when llm.json is absent or incomplete).
#
# HIGH and MID default to the foundation model the paper used, Claude Opus 4.6,
# so that a default run matches the reported setup. The paper's own identifiers
# were `bedrock/eu.anthropic.claude-opus-4-6-v1` on Amazon Bedrock and
# `anthropic/claude-opus-4.6` via OpenRouter; llm.json.example lists both.
#
# LOW is only used for cheap auxiliary calls and is not part of the reported
# configuration.
#
# Model identifiers change. If a default 404s, set `tiers` in llm.json rather
# than editing this file.
DEFAULT_TIERS: Dict[Tier, str] = {
    "HIGH": "anthropic/claude-opus-4-6",
    "MID": "anthropic/claude-opus-4-6",
    "LOW": "anthropic/claude-haiku-4-5",
}

# Default max_tokens per tier (prevents truncation for long outputs)
DEFAULT_TIER_MAX_TOKENS: Dict[Tier, int] = {
    "HIGH": 32768,  # Reports, analysis - generous for long output
    "MID": 16384,   # General use - ReAct agents need room for reasoning
    "LOW": 8192,    # ReAct agents - needs room for tool selection + reasoning
}

# Keys in llm.json that configure this module rather than dspy.LM
_RESERVED_CONFIG_KEYS = {"tiers", "tier_max_tokens"}


def _load_llm_config() -> Dict[str, Any]:
    """Load LLM config from llm.json (optional file)."""
    if not LLM_CONFIG_PATH.exists():
        return {}
    with open(LLM_CONFIG_PATH, "r") as f:
        return json.load(f)


def _resolve_tier_model(config: Dict[str, Any], tier: Tier) -> str:
    """Resolve a tier name to a LiteLLM model string."""
    tiers_config = config.get("tiers", {})
    return tiers_config.get(tier, DEFAULT_TIERS[tier])


def get_lm(
    tier: Tier = "MID",
    model_override: Optional[str] = None,
    **kwargs: Any,
) -> dspy.LM:
    """Return a configured DSPy LM instance.

    Args:
        tier: Capability level - HIGH (complex reasoning), MID (balanced),
              LOW (simple/cost-sensitive). Default: MID.
        model_override: Explicit LiteLLM model string, bypasses tier resolution.
        **kwargs: Additional parameters passed to dspy.LM (e.g., temperature,
                  max_tokens). Explicit max_tokens overrides the tier default.

    Returns:
        Configured dspy.LM instance with tier-appropriate max_tokens.

    Note:
        This function does NOT call dspy.configure(), to avoid global state
        side effects. Callers should use one of:
        1. dspy.settings.context(lm=lm) - for scoped configuration (recommended)
        2. child_predictor.lm = lm - for explicit assignment to the child module
           that makes LLM calls (e.g., self.predict.lm = lm, NOT self.lm = lm).
           DSPy resolves the LM from the calling module, not its parent.
    """
    config = _load_llm_config()

    # Apply the tier default for max_tokens if not explicitly set. Explicit
    # kwargs always win: get_lm(tier="LOW", max_tokens=500) uses 500. A
    # "tier_max_tokens" block in llm.json overrides the built-in defaults.
    if "max_tokens" not in kwargs:
        configured = config.get("tier_max_tokens", {}).get(tier)
        kwargs["max_tokens"] = (
            int(configured) if configured is not None
            else DEFAULT_TIER_MAX_TOKENS.get(tier, 4096)
        )

    # Any non-reserved config key is a dspy.LM kwarg (e.g. api_base, api_key).
    # Explicit kwargs take precedence over the config file.
    for key, value in config.items():
        if key not in _RESERVED_CONFIG_KEYS and key not in kwargs:
            kwargs[key] = value

    model = model_override or _resolve_tier_model(config, tier)
    return dspy.LM(model, **kwargs)


def ensure_llm_ready(tier: Tier = "MID", model_override: Optional[str] = None) -> None:
    """Raise a helpful error if the LM cannot be initialised."""
    try:
        get_lm(tier=tier, model_override=model_override)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise DSPy LM: {exc}\n"
            "Set an API key for your provider (e.g. export ANTHROPIC_API_KEY=...), "
            f"or configure a different model in {LLM_CONFIG_PATH}."
        ) from exc
