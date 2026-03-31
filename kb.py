"""
KnowledgeBase — filesystem abstraction layer for Midas.

Centralises all reads/writes to the knowledge store so the rest of the
codebase never has raw pathlib calls scattered around.

Directory layout (under root_path):
  skills/
      midas-dsl.md
      factor-patterns.md
      evaluation-guide.md
      regime-awareness.md
  knowledge/
      features/
          deployed/       # live features
          candidates/     # passed backtest, awaiting deploy
          archived/       # retired (failure analysis attached)
      learnings/
          offline/        # from backtest iterations
          online/         # from live-trading diagnostics
      regimes/            # market-regime documentation
  proposer/
      prompts/            # plan.md, generate.md, refine.md
  reports/
      daily/
      diagnoses/
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBase:
    """
    Provides typed read/write helpers for all Midas knowledge artefacts.

    Usage:
        kb = KnowledgeBase(Path("./midas-kb"))
        kb.save_offline_learning(doc)
        recent = kb.load_recent_learnings(n=10)
    """

    def __init__(self, root: Path):
        self.root = root
        self._bootstrap()

    # ─────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────

    def _bootstrap(self):
        """Create the full directory tree if it doesn't exist yet."""
        dirs = [
            "skills",
            "knowledge/features/deployed",
            "knowledge/features/candidates",
            "knowledge/features/archived",
            "knowledge/learnings/offline",
            "knowledge/learnings/online",
            "knowledge/regimes",
            "proposer/prompts",
            "reports/daily",
            "reports/diagnoses",
        ]
        for d in dirs:
            (self.root / d).mkdir(parents=True, exist_ok=True)

        # Seed skill files if missing
        self._seed_default_skills()
        self._seed_default_prompts()
        self._seed_default_thresholds()

    # ─────────────────────────────────────────────
    # Skills
    # ─────────────────────────────────────────────

    def load_skill(self, name: str) -> str:
        """Load a skill markdown file (e.g. 'midas-dsl')."""
        path = self.root / "skills" / f"{name}.md"
        if not path.exists():
            return f"[skill '{name}' not found]"
        return path.read_text()

    def load_all_skills(self) -> Dict[str, str]:
        skills: Dict[str, str] = {}
        for f in (self.root / "skills").glob("*.md"):
            skills[f.stem] = f.read_text()
        return skills

    # ─────────────────────────────────────────────
    # Prompts
    # ─────────────────────────────────────────────

    def load_prompt(self, name: str) -> str:
        path = self.root / "proposer" / "prompts" / f"{name}.md"
        if not path.exists():
            return f"[prompt '{name}' not found]"
        return path.read_text()

    # ─────────────────────────────────────────────
    # Features
    # ─────────────────────────────────────────────

    def save_candidate(self, name: str, content: str):
        (self.root / "knowledge" / "features" / "candidates" / f"{name}.md").write_text(content)

    def save_deployed(self, name: str, content: str):
        (self.root / "knowledge" / "features" / "deployed" / f"{name}.md").write_text(content)

    def save_archived(self, name: str, content: str):
        (self.root / "knowledge" / "features" / "archived" / f"{name}.md").write_text(content)

    def load_feature(self, name: str, status: str = "deployed") -> Optional[str]:
        path = self.root / "knowledge" / "features" / status / f"{name}.md"
        return path.read_text() if path.exists() else None

    def list_features(self, status: str = "deployed") -> List[str]:
        folder = self.root / "knowledge" / "features" / status
        return [f.stem for f in folder.glob("*.md")]

    def load_archived_expressions(self, n: int = 20) -> List[str]:
        """Return the raw expression strings from archived features."""
        expressions = []
        folder = self.root / "knowledge" / "features" / "archived"
        for f in sorted(folder.glob("*.md"), reverse=True)[:n]:
            txt = f.read_text()
            # Extract expression block between ``` markers following "expression:"
            m = re.search(r"expression:\s*\|\s*\n(.*?)```", txt, re.DOTALL)
            if m:
                expressions.append(m.group(1).strip())
        return expressions

    # ─────────────────────────────────────────────
    # Learnings
    # ─────────────────────────────────────────────

    def save_offline_learning(self, slug: str, content: str):
        date = datetime.now().strftime("%Y-%m-%d")
        fname = f"{date}-{_slugify(slug)[:40]}.md"
        (self.root / "knowledge" / "learnings" / "offline" / fname).write_text(content)

    def save_online_learning(self, slug: str, content: str):
        date = datetime.now().strftime("%Y-%m-%d")
        fname = f"{date}-{_slugify(slug)[:40]}.md"
        (self.root / "knowledge" / "learnings" / "online" / fname).write_text(content)

    def load_recent_learnings(self, n: int = 10) -> str:
        """Return n most-recent learning docs (both loops) as a single string."""
        docs: List[tuple[datetime, str]] = []

        for folder_name in ("offline", "online"):
            folder = self.root / "knowledge" / "learnings" / folder_name
            for f in folder.glob("*.md"):
                try:
                    date_str = f.name[:10]
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    dt = datetime.min
                docs.append((dt, f.read_text()))

        docs.sort(key=lambda x: x[0], reverse=True)
        return "\n\n---\n\n".join(text for _, text in docs[:n])

    def load_learnings_for_feature(self, feature_name: str, n: int = 10) -> List[str]:
        """Load learnings that mention a specific feature name."""
        results = []
        for folder_name in ("online", "offline"):
            folder = self.root / "knowledge" / "learnings" / folder_name
            for f in sorted(folder.glob("*.md"), reverse=True):
                txt = f.read_text()
                if feature_name.lower() in txt.lower():
                    results.append(txt)
                if len(results) >= n:
                    break
        return results

    # ─────────────────────────────────────────────
    # Regimes
    # ─────────────────────────────────────────────

    def save_regime_doc(self, regime_name: str, content: str):
        fname = f"{_slugify(regime_name)}.md"
        (self.root / "knowledge" / "regimes" / fname).write_text(content)

    def load_regime_docs(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        for f in (self.root / "knowledge" / "regimes").glob("*.md"):
            docs[f.stem] = f.read_text()
        return docs

    # ─────────────────────────────────────────────
    # Reports
    # ─────────────────────────────────────────────

    def save_daily_report(self, date: str, content: str):
        (self.root / "reports" / "daily" / f"{date}.md").write_text(content)

    def save_diagnosis_report(self, slug: str, content: str):
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        (self.root / "reports" / "diagnoses" / f"{ts}-{_slugify(slug)[:30]}.md").write_text(content)

    # ─────────────────────────────────────────────
    # Seeding defaults
    # ─────────────────────────────────────────────

    def _seed_default_skills(self):
        dsl_path = self.root / "skills" / "midas-dsl.md"
        if not dsl_path.exists():
            dsl_path.write_text(_DEFAULT_DSL_SKILL)

        fp_path = self.root / "skills" / "factor-patterns.md"
        if not fp_path.exists():
            fp_path.write_text(_DEFAULT_FACTOR_PATTERNS)

    def _seed_default_prompts(self):
        prompts = {
            "plan":     _DEFAULT_PLAN_PROMPT,
            "generate": _DEFAULT_GENERATE_PROMPT,
            "refine":   _DEFAULT_REFINE_PROMPT,
        }
        for name, content in prompts.items():
            path = self.root / "proposer" / "prompts" / f"{name}.md"
            if not path.exists():
                path.write_text(content)

    def _seed_default_thresholds(self):
        path = self.root / "knowledge" / "thresholds.json"
        if not path.exists():
            thresholds = {
                "min_rank_ic":       0.02,
                "min_ir":            0.50,
                "max_turnover":      0.80,
                "max_correlation":   0.70,
                "max_overfit_ratio": 1.50,
                "min_oos_ic":        0.01,
                "min_composite":     0.30,
            }
            path.write_text(json.dumps(thresholds, indent=2))

    def load_thresholds(self) -> Dict[str, float]:
        path = self.root / "knowledge" / "thresholds.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60]


# ─────────────────────────────────────────────
# Default seeded content
# ─────────────────────────────────────────────

_DEFAULT_DSL_SKILL = """\
# Midas DSL Reference

## Overview
Midas uses a functional DSL for alpha expression definition.
Columns available: open, high, low, close, volume, vwap, trades,
                   bid_volume, ask_volume, open_interest, funding_rate

## Core Operators

### Arithmetic
- `add(x, y)`    — Element-wise addition
- `sub(x, y)`    — Subtraction
- `mul(x, y)`    — Multiplication
- `div(x, y)`    — Division (handles div-by-zero)
- `log(x)`       — Natural log
- `abs(x)`       — Absolute value
- `sign(x)`      — Sign (−1, 0, 1)
- `power(x, n)`  — x^n

### Time Series
- `delay(x, n)`         — Lag by n periods
- `delta(x, n)`         — x − delay(x, n)
- `returns(x, n)`       — (x − delay(x,n)) / delay(x,n)
- `ts_mean(x, n)`       — Rolling mean
- `ts_std(x, n)`        — Rolling std
- `ts_max(x, n)`        — Rolling max
- `ts_min(x, n)`        — Rolling min
- `ts_rank(x, n)`       — Rolling percentile rank [0,1]
- `ts_zscore(x, n)`     — (x − ts_mean) / ts_std
- `ema(x, n)`           — Exponential moving average
- `ts_corr(x, y, n)`    — Rolling Pearson correlation
- `ts_cov(x, y, n)`     — Rolling covariance

### Cross-Sectional
- `cs_rank(x)`           — Cross-sectional percentile rank
- `cs_zscore(x)`         — Cross-sectional z-score
- `cs_demean(x)`         — x − cs_mean(x)
- `cs_neutralize(x, g)`  — Neutralise by group g

### Conditional
- `if_else(cond, x, y)` — Conditional selection
- `clip(x, lo, hi)`     — Clip to range

### Technical
- `rsi(close, n)`                    — Relative Strength Index
- `macd(close, fast, slow, signal)`  — MACD line
- `bbands_pct(close, n, k)`          — BB %B indicator
- `atr(high, low, close, n)`         — Average True Range
- `obv(close, volume)`               — On-Balance Volume

## Constraints
- Max nesting depth : 5
- Max lookback      : 168 bars (1 week at 1-hour bars)
- Output must be stationary (use returns/zscore/rank — not raw price)
- No look-ahead: all lookbacks must be positive integers

## Example Expressions

```
# Momentum: VWAP-normalised price move
div(sub(close, ts_mean(vwap, 48)), atr(high, low, close, 24))

# Mean reversion: RSI inversion
sub(50, rsi(close, 14))

# Order-book imbalance
div(sub(bid_volume, ask_volume), add(bid_volume, ask_volume))

# Volume-adjusted momentum
mul(returns(close, 24), ts_zscore(volume, 48))
```
"""

_DEFAULT_FACTOR_PATTERNS = """\
# Common Alpha Factor Patterns

## Momentum
- Short-term price momentum (1–8h)
- VWAP deviation momentum
- Volume-weighted price momentum

## Mean Reversion
- RSI extremes
- Bollinger Band %B
- Z-score of returns vs rolling baseline

## Microstructure
- Bid/ask imbalance
- Trade-flow imbalance (taker buy vs sell)
- Volume surprise (actual vs expected)

## Carry / Funding
- Perpetual funding rate as carry signal
- Open-interest change as positioning signal

## Regime Filters
- Use `if_else` to gate signal on ATR regime
- Separate bull/bear factor weights via regime label
"""

_DEFAULT_PLAN_PROMPT = """\
# Alpha Feature Planning Prompt

You are a senior quant researcher planning a new alpha feature for crypto perp futures.

## Context
- Data schema        : {{data_schema}}
- Existing factors   : {{existing_factors}}
- Recent learnings   : {{recent_learnings}}
- Current regime     : {{current_regime}}

## Task
Generate a research plan for a new alpha feature that:
1. Is NOT highly correlated with existing factors
2. Has clear economic intuition for crypto perp markets
3. Is appropriate for the current market regime
4. Avoids patterns that have previously failed (see learnings above)

## Output (YAML only, no prose)
```yaml
hypothesis: |
  Economic intuition for why this should predict returns

target_horizon: "1h" | "4h" | "24h"

data_sources:
  - field: <column_name>
    rationale: <how it will be used>

expression_sketch: |
  Rough DSL idea — not final

risks:
  - <potential failure mode>

related_learnings:
  - <references to past learnings>
```
"""

_DEFAULT_GENERATE_PROMPT = """\
# Alpha Expression Generation Prompt

You are a senior quant researcher writing Midas DSL alpha expressions.

## DSL Reference
{{midas_dsl_skill}}

## Research Plan
{{plan_output}}

## Previously Failed Expressions (avoid similar patterns)
{{failed_expressions}}

## Rules
- Valid Midas DSL only
- No future-data leakage (all lookbacks > 0)
- Output must be stationary (zscore / returns / rank)
- Max nesting depth: 5
- Simpler is better — complex expressions overfit

## Output (YAML only)
```yaml
candidates:
  - name: "descriptive_snake_case_name"
    expression: |
      the_dsl_expression_here
    rationale: "why this specific formulation"

  - name: "..."
    expression: |
      ...
    rationale: "..."

  - name: "..."
    expression: |
      ...
    rationale: "..."
```
"""

_DEFAULT_REFINE_PROMPT = """\
# Alpha Expression Refinement Prompt

You are iterating on an alpha feature based on multi-agent evaluation feedback.

## Original Expression
{{expression}}

## Evaluation Results (JSON)
{{evaluation_result}}

## Blocking Issues
{{blocking_issues}}

## Agent Suggestions
{{suggestions}}

## Common Fixes
- High turnover          → wrap in ema() to smooth
- Short half-life        → increase lookback or add lag
- High existing corr     → residualise with cs_neutralize or sub out the correlated component
- Regime-dependent       → add if_else regime filter
- Overfit (ratio > 1.5)  → simplify — remove parameters or reduce nesting

## Output (YAML only)
```yaml
refined_expression: |
  the_improved_dsl_expression

changes_made:
  - "<change> — <reason>"

expected_improvement:
  - "<metric> should improve because <reason>"
```
"""
