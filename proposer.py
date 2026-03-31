"""
Midas Proposer
==============
Two components:

  DSLValidator      — lightweight syntax/structure check for Midas DSL
                      (no LLM, runs in the inner loop)

  ExpressionProposer — LLM-backed generator that produces Plan → candidates,
                       and Refine → improved candidates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .kb import KnowledgeBase
from .models import LearningDocument, MultiAgentResult


# ─────────────────────────────────────────────────────────────────────────────
# DSL Validator
# ─────────────────────────────────────────────────────────────────────────────

# All legal operators in the Midas DSL
_VALID_OPS = {
    # arithmetic
    "add", "sub", "mul", "div", "log", "abs", "sign", "power",
    # time-series
    "delay", "delta", "returns", "ts_mean", "ts_std", "ts_max", "ts_min",
    "ts_rank", "ts_zscore", "ema", "ts_corr", "ts_cov",
    # cross-sectional
    "cs_rank", "cs_zscore", "cs_demean", "cs_neutralize",
    # conditional
    "if_else", "clip", "max", "min",
    # technical
    "rsi", "macd", "bbands_pct", "atr", "obv",
}

# Columns available in the trading data schema
_VALID_COLUMNS = {
    "open", "high", "low", "close", "volume", "vwap", "trades",
    "bid_volume", "ask_volume", "open_interest", "funding_rate",
}


@dataclass
class ValidationResult:
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self):
        return self.valid


class DSLValidator:
    """
    Validates a Midas DSL expression string before sending it to the
    feature engine.  Catches common LLM hallucinations early.

    Checks:
      1. Parentheses balanced
      2. No unknown operators
      3. No unknown column names
      4. Nesting depth ≤ 5
      5. No obvious future-leakage patterns (negative lookbacks)
    """

    def validate(self, expression: str) -> ValidationResult:
        errors:   List[str] = []
        warnings: List[str] = []

        expr = expression.strip()

        if not expr:
            return ValidationResult(False, ["Empty expression"])

        # 1. Balanced parentheses
        depth = 0
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                errors.append("Unbalanced parentheses (too many closing)")
                break
        if depth != 0:
            errors.append(f"Unbalanced parentheses (unclosed: {depth})")

        # 2. Unknown operators
        found_ops = set(re.findall(r"\b([a-z_]+)\s*\(", expr))
        unknown   = found_ops - _VALID_OPS
        if unknown:
            errors.append(f"Unknown operators: {unknown}")

        # 3. Unknown columns
        # Only flag standalone words not preceded by "(" (i.e. column refs)
        tokens     = set(re.findall(r"\b([a-z_]+)\b", expr))
        candidates = tokens - _VALID_OPS - {"true", "false", "null"}
        # Remove numeric-like tokens
        candidates = {t for t in candidates if not re.match(r"^\d+$", t)}
        bad_cols   = candidates - _VALID_COLUMNS
        if bad_cols:
            warnings.append(f"Unrecognised tokens (may be column aliases): {bad_cols}")

        # 4. Nesting depth
        max_depth = 0
        cur_depth = 0
        for ch in expr:
            if ch == "(":
                cur_depth += 1
                max_depth = max(max_depth, cur_depth)
            elif ch == ")":
                cur_depth -= 1
        if max_depth > 5:
            errors.append(f"Nesting depth {max_depth} exceeds limit of 5")

        # 5. Negative lookbacks (quick heuristic)
        neg_lookbacks = re.findall(r",\s*(-\d+)\s*\)", expr)
        if neg_lookbacks:
            errors.append(f"Negative lookbacks detected {neg_lookbacks} — potential look-ahead bias")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# ExpressionProposer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    name:       str
    expression: str
    rationale:  str


@dataclass
class ResearchPlan:
    hypothesis:        str
    target_horizon:    str
    data_sources:      List[Dict[str, str]]
    expression_sketch: str
    risks:             List[str]
    related_learnings: List[str]
    raw:               Dict[str, Any] = field(default_factory=dict)


class ExpressionProposer:
    """
    LLM-backed alpha expression generator.

    Phase 1 — Plan:
        generate_plan(goal, existing_factors, regime) → ResearchPlan

    Phase 2 — Generate:
        generate_candidates(plan, kb) → List[Candidate]

    Phase 3 — Refine:
        refine(expression, result, kb) → List[Candidate]
    """

    def __init__(
        self,
        kb:         KnowledgeBase,
        llm_client,                 # anthropic.Anthropic instance
        model:      str = "claude-sonnet-4-20250514",
        max_tokens: int = 2_000,
    ):
        self.kb         = kb
        self.llm        = llm_client
        self.model      = model
        self.max_tokens = max_tokens
        self.validator  = DSLValidator()

    # ─────────────────────────────────────────────
    # Plan
    # ─────────────────────────────────────────────

    def generate_plan(
        self,
        research_goal:    str,
        existing_factors: List[str],
        regime:           str        = "UNKNOWN",
        data_schema:      str        = "",
    ) -> ResearchPlan:
        """Generate a structured research plan for a new alpha feature."""
        template = self.kb.load_prompt("plan")
        recent   = self.kb.load_recent_learnings(n=8)

        prompt = template.replace("{{data_schema}}",     data_schema or _DEFAULT_SCHEMA)
        prompt = prompt.replace("{{existing_factors}}",  json.dumps(existing_factors))
        prompt = prompt.replace("{{recent_learnings}}",  recent or "None yet.")
        prompt = prompt.replace("{{current_regime}}",    regime)

        raw = self._call_llm(f"{research_goal}\n\n{prompt}")
        parsed = _parse_yaml(raw)

        return ResearchPlan(
            hypothesis        = parsed.get("hypothesis", ""),
            target_horizon    = parsed.get("target_horizon", "1h"),
            data_sources      = parsed.get("data_sources", []),
            expression_sketch = parsed.get("expression_sketch", ""),
            risks             = parsed.get("risks", []),
            related_learnings = parsed.get("related_learnings", []),
            raw               = parsed,
        )

    # ─────────────────────────────────────────────
    # Generate
    # ─────────────────────────────────────────────

    def generate_candidates(
        self,
        plan: ResearchPlan,
    ) -> List[Candidate]:
        """Generate 3 diverse candidate expressions from a research plan."""
        template = self.kb.load_prompt("generate")
        dsl_ref  = self.kb.load_skill("midas-dsl")
        failed   = self.kb.load_archived_expressions(n=20)

        prompt = template.replace("{{midas_dsl_skill}}",   dsl_ref)
        prompt = prompt.replace("{{plan_output}}",          json.dumps(plan.raw, indent=2))
        prompt = prompt.replace("{{failed_expressions}}",   "\n".join(failed) or "None yet.")

        raw     = self._call_llm(prompt)
        parsed  = _parse_yaml(raw)
        return self._extract_candidates(parsed)

    # ─────────────────────────────────────────────
    # Refine
    # ─────────────────────────────────────────────

    def refine(
        self,
        expression: str,
        result:     MultiAgentResult,
    ) -> List[Candidate]:
        """Refine a failing expression based on multi-agent feedback."""
        template = self.kb.load_prompt("refine")

        prompt = template.replace("{{expression}}",        expression)
        prompt = prompt.replace("{{evaluation_result}}",   json.dumps(result.metrics.to_dict(), indent=2))
        prompt = prompt.replace("{{blocking_issues}}",     "\n".join(f"- {b}" for b in result.blocking_issues))
        prompt = prompt.replace("{{suggestions}}",         "\n".join(f"- {s}" for s in result.improvement_suggestions))

        raw    = self._call_llm(prompt)
        parsed = _parse_yaml(raw)

        refined = parsed.get("refined_expression", "")
        if not refined:
            return []

        name = f"refined_{len(result.blocking_issues)}issues"
        return [Candidate(name=name, expression=refined, rationale=str(parsed.get("changes_made", "")))]

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _extract_candidates(self, parsed: Dict[str, Any]) -> List[Candidate]:
        raw_list = parsed.get("candidates", [])
        out: List[Candidate] = []
        for item in raw_list:
            expr = str(item.get("expression", "")).strip()
            if not expr:
                continue
            out.append(Candidate(
                name      = item.get("name", "unnamed"),
                expression= expr,
                rationale = item.get("rationale", ""),
            ))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yaml(text: str) -> Dict[str, Any]:
    """Extract and parse the first YAML block in an LLM response."""
    # Strip fenced code blocks
    if "```yaml" in text:
        text = text.split("```yaml")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


_DEFAULT_SCHEMA = (
    "open, high, low, close, volume, vwap, trades, "
    "bid_volume, ask_volume, open_interest, funding_rate"
)
