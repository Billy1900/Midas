"""
Midas Offline Compound Loop
============================
Implements the discovery arm of the dual-loop system.

Flow:
  Plan → Write (generate) → Assess (multi-agent) → Learn
            ↑                                         |
            └──────────── Refine ←────────────────────┘
                         (if failing)

The key compound-engineering property: every loop run writes a structured
LearningDocument to the knowledge base.  Future runs load those docs in the
Plan phase, so the LLM can avoid repeating past mistakes and build on past
successes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

from .evaluator  import AlphaEvaluator, MultiAgentEvaluator
from .kb         import KnowledgeBase, _slugify
from .models     import LearningDocument, LoopIteration, MultiAgentResult
from .proposer   import Candidate, DSLValidator, ExpressionProposer

logger = logging.getLogger("midas.offline")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OfflineLoopConfig:
    """Tunable knobs for the offline loop."""
    max_iterations:      int   = 10
    candidates_per_iter: int   = 3    # expressions generated per LLM call
    model:               str   = "claude-sonnet-4-20250514"
    max_tokens:          int   = 2_000
    verbose:             bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Offline loop
# ─────────────────────────────────────────────────────────────────────────────

class OfflineCompoundLoop:
    """
    Orchestrates the Plan → Write → Assess → Learn loop.

    Usage:
        loop = OfflineCompoundLoop(kb, evaluator, proposer, cfg)
        learning = loop.run(
            research_goal    = "Find a short-term mean-reversion signal",
            existing_factors = ["momentum_ema_cross"],
            data_fn          = my_data_provider,   # callable → (feature_values, fwd_rets, regimes)
            regime           = "HIGH_VOL",
        )
    """

    def __init__(
        self,
        kb:        KnowledgeBase,
        evaluator: MultiAgentEvaluator,
        proposer:  ExpressionProposer,
        config:    OfflineLoopConfig = None,
        on_iteration: Optional[Callable[[LoopIteration], None]] = None,
    ):
        self.kb           = kb
        self.evaluator    = evaluator
        self.proposer     = proposer
        self.cfg          = config or OfflineLoopConfig()
        self.on_iteration = on_iteration
        self.validator    = DSLValidator()

    # ─────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────

    def run(
        self,
        research_goal:    str,
        existing_factors: List[str],
        data_fn:          Callable[[], tuple],   # () → (feature_values_fn, fwd_returns, regimes)
        regime:           str = "UNKNOWN",
        data_schema:      str = "",
    ) -> LearningDocument:
        """
        Execute one full research session.

        Args:
            research_goal    : Natural-language description of the desired signal.
            existing_factors : Names of currently deployed factors.
            data_fn          : Zero-arg callable returning
                               (compute_fn, forward_returns_df, regime_series).
                               compute_fn takes an expression string and returns
                               a pd.Series of feature values.
            regime           : Current market regime label.
            data_schema      : Optional data column descriptions for the planner.

        Returns:
            LearningDocument (saved to the knowledge base).
        """
        self._log(f"=== Offline Loop START | goal: {research_goal!r} ===")
        iterations: List[LoopIteration] = []

        # Unpack data
        compute_fn, fwd_returns, regime_series = data_fn()

        # ── Phase 1: PLAN ────────────────────────────────────────────────────
        self._log("Phase: PLAN")
        plan = self.proposer.generate_plan(
            research_goal    = research_goal,
            existing_factors = existing_factors,
            regime           = regime,
            data_schema      = data_schema,
        )
        self._log(f"  hypothesis: {plan.hypothesis[:120]}")

        # ── Phases 2–4: WRITE → ASSESS → LEARN (iterate) ───────────────────
        for i in range(self.cfg.max_iterations):
            self._log(f"\nIteration {i + 1}/{self.cfg.max_iterations}")

            # WRITE
            if i == 0 or not iterations:
                # First attempt, or all previous iterations produced no scoreable candidates
                # — re-generate fresh from the plan rather than refining nothing
                candidates = self.proposer.generate_candidates(plan)
            else:
                last = iterations[-1]
                candidates = self.proposer.refine(last.expression, last.evaluation)
                if not candidates:
                    self._log("  Refine returned no candidates — falling back to generate")
                    candidates = self.proposer.generate_candidates(plan)

            # Evaluate all candidates, keep best
            best_result:     Optional[MultiAgentResult] = None
            best_expression: str = ""

            for cand in candidates:
                self._log(f"  Candidate: {cand.name}")

                # DSL validation (fast, no LLM)
                v = self.validator.validate(cand.expression)
                if not v.valid:
                    self._log(f"    ✗ DSL invalid: {v.errors}")
                    continue
                if v.warnings:
                    self._log(f"    ⚠ DSL warnings: {v.warnings}")

                # Feature computation
                try:
                    feature_values = compute_fn(cand.expression)
                    feature_values.name = cand.name
                except Exception as e:
                    self._log(f"    ✗ Compute failed: {e}")
                    continue

                # ASSESS
                result = self.evaluator.evaluate(feature_values, fwd_returns, regime_series)

                score = result.metrics.composite_score
                self._log(f"    score={score:.2f}  pass={result.overall_pass}")
                self._log(f"    {result.metrics.summary()}")

                if best_result is None or score > best_result.metrics.composite_score:
                    best_result     = result
                    best_expression = cand.expression

            if best_result is None:
                self._log("  No valid candidates this iteration")
                continue

            # Determine action
            if best_result.overall_pass:
                action = "accept"
            elif i < self.cfg.max_iterations - 1:
                action = "refine"
            else:
                action = "reject"

            feedback = _format_feedback(best_result)

            iteration = LoopIteration(
                iteration  = i,
                expression = best_expression,
                evaluation = best_result,
                feedback   = feedback,
                action     = action,
            )
            iterations.append(iteration)

            if self.on_iteration:
                self.on_iteration(iteration)

            if action == "accept":
                self._log(f"  ✓ ACCEPTED at iteration {i + 1}")
                break
            else:
                self._log(f"  → {action.upper()}")
                if best_result.blocking_issues:
                    for issue in best_result.blocking_issues:
                        self._log(f"    blocking: {issue}")

        # ── Phase 4: LEARN ───────────────────────────────────────────────────
        self._log("\nPhase: LEARN")
        learning = self._generate_learning(research_goal, iterations)
        self._save_learning(learning)

        if iterations and iterations[-1].action == "accept":
            self._save_candidate(iterations[-1].expression, learning)

        self._log("=== Offline Loop END ===\n")
        return learning

    # ─────────────────────────────────────────────
    # Learning generation
    # ─────────────────────────────────────────────

    def _generate_learning(
        self,
        research_goal: str,
        iterations:    List[LoopIteration],
    ) -> LearningDocument:
        """Use the LLM to synthesise a learning document from iteration history."""

        if not iterations:
            return LearningDocument(
                date                    = _today(),
                expression              = "",
                result                  = "failed",
                metrics                 = {},
                why_it_worked_or_failed = "No valid expressions were generated",
                pattern_identified      = "DSL syntax or data alignment issue",
                suggestions_for_future  = ["Verify DSL expressions against schema before submitting"],
                related_learnings       = [],
            )

        final = iterations[-1]

        # Serialise iteration history (drop heavy DataFrames)
        history = [
            {
                "iteration":  it.iteration,
                "expression": it.expression,
                "action":     it.action,
                "blocking":   it.evaluation.blocking_issues,
                "score":      it.evaluation.metrics.composite_score,
                "rank_ic":    it.evaluation.metrics.rank_ic,
                "ir":         it.evaluation.metrics.ir,
            }
            for it in iterations
        ]

        prompt = f"""Analyse this alpha research session and produce a concise learning document.

Research Goal: {research_goal}

Iteration History:
{json.dumps(history, indent=2)}

Final Outcome: {'SUCCESS' if final.action == 'accept' else 'FAILED'}

Write the following YAML (no extra prose):

```yaml
why_it_worked_or_failed: |
  Specific technical reason

pattern_identified: |
  Single-sentence generalizable insight (used as document title)

suggestions_for_future:
  - actionable item 1
  - actionable item 2

related_learnings:
  - reference to any prior pattern or known issue
```
"""
        try:
            response = self.proposer.llm.messages.create(
                model      = self.cfg.model,
                max_tokens = 1_000,
                messages   = [{"role": "user", "content": prompt}],
            )
            parsed = _parse_yaml_text(response.content[0].text)
        except Exception as e:
            logger.warning(f"Learning LLM call failed: {e}")
            parsed = {}

        return LearningDocument(
            date                    = _today(),
            expression              = final.expression,
            result                  = "success" if final.action == "accept" else "failed",
            metrics                 = {
                k: v for k, v in final.evaluation.metrics.to_dict().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            why_it_worked_or_failed = parsed.get("why_it_worked_or_failed", ""),
            pattern_identified      = parsed.get("pattern_identified", "Unknown pattern"),
            suggestions_for_future  = parsed.get("suggestions_for_future", []),
            related_learnings       = parsed.get("related_learnings", []),
        )

    def _save_learning(self, learning: LearningDocument):
        slug = _slugify(learning.pattern_identified)
        self.kb.save_offline_learning(slug, learning.to_markdown())
        self._log(f"  Learning saved: {slug}")

    def _save_candidate(self, expression: str, learning: LearningDocument):
        name = _slugify(learning.pattern_identified)[:40]
        m = learning.metrics
        content = f"""# Candidate Feature: {name}

**status:** candidate  |  **discovered:** {learning.date}  |  **score:** {m.get('composite_score', 0):.2f}

---

## Expression

```
{expression}
```

## Evaluation Summary

| metric        | value |
|---------------|-------|
| rank_ic       | {m.get('rank_ic', 0):.4f} |
| ir            | {m.get('ir', 0):.2f} |
| turnover      | {m.get('turnover', 0):.2f} |
| oos_ic        | {m.get('oos_ic', 0):.4f} |
| half_life_h   | {m.get('half_life_hours', float('nan')):.1f} |
| composite     | {m.get('composite_score', 0):.2f} |

## Economic Rationale

{learning.why_it_worked_or_failed}

## Deployment Notes

{chr(10).join(f'- {s}' for s in learning.suggestions_for_future[:3])}
"""
        self.kb.save_candidate(name, content)
        self._log(f"  Candidate saved: {name}")

    # ─────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────

    def _log(self, msg: str):
        if self.cfg.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")
        logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_feedback(result: MultiAgentResult) -> str:
    lines = [
        f"Composite Score : {result.metrics.composite_score:.2f}",
        f"Overall         : {'PASS ✓' if result.overall_pass else 'FAIL ✗'}",
        "",
        "Agent Verdicts:",
        result.verdict_table(),
    ]
    if result.blocking_issues:
        lines += ["", "Blocking Issues:"] + [f"  • {b}" for b in result.blocking_issues]
    if result.improvement_suggestions:
        lines += ["", "Suggestions:"] + [f"  → {s}" for s in result.improvement_suggestions[:4]]
    return "\n".join(lines)


def _parse_yaml_text(text: str) -> Dict[str, Any]:
    if "```yaml" in text:
        text = text.split("```yaml")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")
