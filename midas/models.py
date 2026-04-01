"""
Shared data models for the Midas framework.
All dataclasses used across multiple modules live here to avoid circular imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────
# Evaluator models
# ─────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Comprehensive metrics for a single alpha feature."""

    feature_name: str

    # --- Predictive power ---
    ic:       float = 0.0    # Pearson IC vs 1-hour forward return
    rank_ic:  float = 0.0    # Spearman rank IC (more robust)
    ic_std:   float = 0.0    # Rolling IC standard deviation
    ir:       float = 0.0    # Information Ratio = IC / IC_std

    # --- Decay analysis ---
    ic_decay_curve:  List[float] = field(default_factory=list)  # IC at [1h,4h,8h,24h,48h]
    half_life_hours: float = float("nan")                        # Hours to IC = 0.5 * IC_0

    # --- Trading-cost proxy ---
    turnover:     float = 0.0  # Mean abs daily position change
    effective_ic: float = 0.0  # IC − (turnover × spread_cost)

    # --- Diversification ---
    correlation_with_existing: Dict[str, float] = field(default_factory=dict)
    max_correlation: float = 0.0
    marginal_ic:     float = 0.0  # IC orthogonal to existing factors

    # --- Robustness ---
    ic_by_regime:      Dict[str, float] = field(default_factory=dict)
    ic_quintile_spread: float = 0.0
    sharpe_quintile:    float = 0.0

    # --- Out-of-sample ---
    is_ic:         float = 0.0
    oos_ic:        float = 0.0
    overfit_ratio: float = float("inf")

    # --- Composite ---
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.feature_name}] "
            f"rankIC={self.rank_ic:.4f}  IR={self.ir:.2f}  "
            f"HL={self.half_life_hours:.1f}h  OOS={self.oos_ic:.4f}  "
            f"maxCorr={self.max_correlation:.2f}  score={self.composite_score:.2f}"
        )


@dataclass
class AgentVerdict:
    """A single evaluation agent's assessment."""
    agent_name:  str
    passed:      bool
    score:       float
    details:     str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class MultiAgentResult:
    """Combined verdict from all evaluation agents."""
    feature_name:            str
    metrics:                 EvaluationResult
    verdicts:                List[AgentVerdict]
    overall_pass:            bool
    blocking_issues:         List[str]
    improvement_suggestions: List[str]

    def verdict_table(self) -> str:
        rows = []
        for v in self.verdicts:
            sym = "✓" if v.passed else "✗"
            rows.append(f"  {sym} {v.agent_name:<25} {v.details}")
        return "\n".join(rows)


# ─────────────────────────────────────────────
# Offline-loop models
# ─────────────────────────────────────────────

@dataclass
class LoopIteration:
    """State of a single Plan→Write→Assess iteration."""
    iteration:  int
    expression: str
    evaluation: MultiAgentResult
    feedback:   str
    action:     str   # "refine" | "accept" | "reject"


@dataclass
class LearningDocument:
    """Structured knowledge artefact written after every loop run."""
    date:                    str
    expression:              str
    result:                  str   # "success" | "failed"
    metrics:                 Dict[str, Any]
    why_it_worked_or_failed: str
    pattern_identified:      str
    suggestions_for_future:  List[str]
    related_learnings:       List[str]

    def to_markdown(self) -> str:
        sug = "\n".join(f"- {s}" for s in self.suggestions_for_future)
        rel = "\n".join(f"- {r}" for r in self.related_learnings)
        return f"""# Offline Learning: {self.pattern_identified}

**date:** {self.date}  |  **result:** {self.result}

---

## Expression

```
{self.expression}
```

## Why It {'Worked' if self.result == 'success' else 'Failed'}

{self.why_it_worked_or_failed}

## Pattern Identified

{self.pattern_identified}

## Suggestions for Future

{sug}

## Key Metrics

```json
{json.dumps({k: v for k, v in self.metrics.items()
             if isinstance(v, (int, float, str, bool, type(None)))}, indent=2)}
```

## Related Learnings

{rel}
"""


# ─────────────────────────────────────────────
# Online-loop models
# ─────────────────────────────────────────────

class AlertType(Enum):
    IC_DECAY             = "ic_decay"
    REGIME_SHIFT         = "regime_shift"
    EXECUTION_SLIPPAGE   = "execution_slippage"
    CORRELATION_SPIKE    = "correlation_spike"
    DRAWDOWN             = "drawdown"


@dataclass
class Alert:
    """A triggered monitoring alert."""
    alert_type:         AlertType
    feature_name:       str
    timestamp:          datetime
    severity:           str   # "warning" | "critical"
    details:            Dict[str, Any]
    threshold_breached: str


@dataclass
class FeatureMetrics:
    """Rolling runtime metrics for a single deployed feature."""
    feature_name:        str
    timestamp:           datetime

    # Rolling ICs
    ic_1d:    float = float("nan")
    ic_5d:    float = float("nan")
    ic_30d:   float = float("nan")
    ic_ratio: float = float("nan")   # ic_5d / ic_30d

    # Decay
    half_life_hours: float = float("nan")

    # Execution
    realized_turnover:     float = 0.0
    slippage_vs_expected:  float = 0.0

    # Attribution
    pnl_contribution_bps: float = 0.0

    # Regime
    current_regime: str   = "UNKNOWN"
    ic_in_regime:   float = float("nan")


@dataclass
class DailyReport:
    """End-of-day summary produced by OnlineMonitor."""
    date:              str
    portfolio_metrics: Dict[str, Any]
    feature_breakdown: List[FeatureMetrics]
    alerts_triggered:  List[Alert]
    regime_info:       Dict[str, Any]
    execution_summary: Dict[str, Any]


@dataclass
class DiagnoseResult:
    """Output of the LLM-powered diagnose agent."""
    feature_name: str
    alert:        Alert
    root_cause:   str
    evidence:     List[str]
    fix_proposal: Optional[str]
    kill_signal:  bool
    learning_doc: str
