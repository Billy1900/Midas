"""
Midas Integration Test & Demo
==============================
Runs a complete end-to-end test of both the offline and online loops
using a mock LLM client — no real Anthropic API key required.

Run with:
    python tests/test_integration.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add parent to path so we can import midas directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from midas.evaluator  import AlphaEvaluator, MultiAgentEvaluator
from midas.kb         import KnowledgeBase
from midas.loops      import OfflineCompoundLoop, OfflineLoopConfig
from midas.models     import MultiAgentResult
from midas.monitor    import OnlineMonitor
from midas.promoter   import FeaturePromoter
from midas.proposer   import Candidate, DSLValidator, ExpressionProposer


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM
# ─────────────────────────────────────────────────────────────────────────────

class MockMessage:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(text=text)]

class MockLLM:
    """Returns plausible YAML for every request."""

    def __init__(self):
        self.call_count = 0

    def create(self, **kwargs) -> MockMessage:
        self.call_count += 1
        content = kwargs.get("messages", [{}])[-1].get("content", "")

        # Most-specific checks first — each anchored to a unique phrase
        # that only appears in that prompt type (not in substituted content).

        # 1. Diagnosis — unique anchor: "kill_signal"
        if "kill_signal" in content.lower() or "root_cause" in content.lower():
            return MockMessage("""```yaml
root_cause: |
  The IC ratio has dropped because the feature relies on VWAP deviation
  which became less predictive during the recent low-volatility grinding
  regime. Mean-reversion signals typically underperform during persistent
  directional trends.

evidence:
  - ic_5d has fallen from 0.032 to 0.008 over 3 weeks
  - The market has been in LOW_VOL regime for 18 consecutive days
  - Turnover has increased, suggesting signal is flipping rapidly

fix_proposal: |
  ema(ts_zscore(div(sub(close, vwap), atr(high, low, close, 48)), 96), 6)

kill_signal: false

learning_doc: |
  # Online Learning: VWAP mean-reversion degrades in low-vol trending regime

  ## Observation
  IC_5d / IC_30d ratio dropped below 0.5 (critical threshold)

  ## Root Cause
  Low-vol trending regimes suppress mean-reversion dynamics

  ## Pattern Identified
  Add a regime filter: gate VWAP mean-rev signal off when ATR < 20th percentile

  ## Action Taken
  Extended lookback from 24→48 and added EMA smoothing; monitoring for recovery

  ## Suggestions for Future
  - Always include regime-conditional filter for mean-rev signals
  - Monitor ATR percentile as a proxy for regime suitability
```""")

        # 2. Learning synthesis — unique anchor: "iteration history"
        if "iteration history" in content.lower():
            return MockMessage("""```yaml
why_it_worked_or_failed: |
  VWAP-deviation normalised by ATR produced consistent mean-reversion
  alpha across regimes because it captures normalised over-extension
  relative to fair value, which is robust to vol-regime changes.

pattern_identified: |
  ATR-normalised VWAP deviation is a robust mean-reversion signal

suggestions_for_future:
  - Apply ema smoothing with span 4-8 to improve cost efficiency
  - Test with cs_rank() wrapper for better cross-sectional neutrality

related_learnings:
  - Raw price minus VWAP without normalisation fails in high-vol regimes
```""")

        # 3. Refine — unique anchor: "blocking issues"
        if "blocking issues" in content.lower():
            return MockMessage("""```yaml
refined_expression: |
  ts_zscore(ema(div(sub(close, vwap), atr(high, low, close, 24)), 4), 48)

changes_made:
  - "Wrapped raw signal in ema(4) to reduce turnover by ~40%"

expected_improvement:
  - "turnover should drop below 0.80 threshold"
  - "effective_ic should improve due to lower cost drag"
```""")

        # 4. Generate candidates — unique anchor: "previously fail" (from template header)
        if "previously fail" in content.lower() or "alpha expression generation" in content.lower():
            return MockMessage("""```yaml
candidates:
  - name: "vwap_zscore_mean_rev"
    expression: |
      ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)
    rationale: "VWAP deviation normalised by ATR — captures mean-reversion pressure"

  - name: "rsi_inversion"
    expression: |
      sub(50, rsi(close, 14))
    rationale: "RSI inversion: long when oversold, short when overbought"

  - name: "volume_imbalance_smooth"
    expression: |
      ema(div(sub(bid_volume, ask_volume), add(bid_volume, ask_volume)), 8)
    rationale: "Smoothed order-book imbalance captures short-term directional pressure"
```""")

        # 5. Plan — unique anchor: "alpha feature planning"
        if "alpha feature planning" in content.lower() or "expression_sketch" in content.lower():
            return MockMessage("""```yaml
hypothesis: |
  When the current close price is significantly above the VWAP over a
  rolling 24-bar window, the asset tends to mean-revert downward in the
  subsequent hour due to order-book pressure.

target_horizon: "1h"

data_sources:
  - field: close
    rationale: used to compute deviation from VWAP
  - field: vwap
    rationale: fair-value benchmark
  - field: atr
    rationale: normalise the deviation

expression_sketch: |
  ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)

risks:
  - may underperform in strong trending regimes
  - high turnover if raw signal is noisy

related_learnings: []
```""")

        # fallback
        return MockMessage("```yaml\nresult: ok\n```")

    @property
    def messages(self):
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_data(n: int = 500, seed: int = 42) -> tuple:
    """
    Generate plausible synthetic feature + return data for testing.

    Returns:
        compute_fn      : callable(expr_str) → pd.Series
        forward_returns : pd.DataFrame with ret_1h, ret_4h, ret_8h, ret_24h, ret_48h
        regimes         : pd.Series of regime labels
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    # True alpha signal: VWAP deviation with noise
    true_signal = rng.normal(0, 1, n)
    true_signal = pd.Series(true_signal, index=idx, name="true_signal")

    # Forward returns correlated with true signal
    noise  = rng.normal(0, 5, n)
    ret_1h = true_signal * 0.03 + noise * 0.01
    fwd_returns = pd.DataFrame({
        "ret_1h":  ret_1h,
        "ret_4h":  true_signal * 0.025 + rng.normal(0, 1, n) * 0.015,
        "ret_8h":  true_signal * 0.020 + rng.normal(0, 1, n) * 0.020,
        "ret_24h": true_signal * 0.010 + rng.normal(0, 1, n) * 0.030,
        "ret_48h": true_signal * 0.005 + rng.normal(0, 1, n) * 0.040,
    }, index=idx)

    # Regime labels
    regimes = pd.Series(
        rng.choice(["HIGH_VOL", "LOW_VOL", "TRENDING"], n),
        index=idx, name="regime",
    )

    def compute_fn(expr: str) -> pd.Series:
        """
        Mock compute function.
        Returns the true signal with expression-name-based perturbation
        so different expressions get slightly different scores.
        """
        perturbation = rng.normal(0, 0.3, n)
        if "zscore" in expr or "mean_rev" in expr.lower():
            signal = -true_signal * 0.8 + perturbation   # mean-rev
        elif "rsi" in expr:
            signal = -true_signal * 0.6 + perturbation
        elif "imbalance" in expr or "bid" in expr:
            signal = true_signal * 0.5 + perturbation
        else:
            signal = true_signal * 0.4 + perturbation
        s = pd.Series(signal, index=idx)
        s.name = expr[:30]
        return s

    def data_fn():
        return compute_fn, fwd_returns, regimes

    return data_fn, fwd_returns, regimes


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


def test_dsl_validator():
    _section("Test 1 — DSL Validator")
    from midas.proposer import DSLValidator
    v = DSLValidator()

    cases = [
        ("ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)", True),
        ("ts_zscore(div(sub(close, vwap), atr(high, low, close, -24)), 48)", False),  # neg lookback
        ("magic_op(close, 10)", False),  # unknown op
        ("",                            False),  # empty
        ("add(sub(mul(div(ema(close, 5), vwap), ts_mean(volume, 10)), 1.0), ts_rank(close, 20))", True),
    ]

    for expr, expected in cases:
        result = v.validate(expr)
        status = "✓" if result.valid == expected else "✗ FAIL"
        print(f"  {status}  valid={result.valid}  expr={expr[:50]!r}")
        if result.errors:
            print(f"       errors: {result.errors}")
    print()


def test_evaluator(fwd_returns: pd.DataFrame, regimes: pd.Series):
    _section("Test 2 — Static Evaluator")

    rng    = np.random.default_rng(0)
    n      = len(fwd_returns)
    signal = pd.Series(
        -fwd_returns["ret_1h"].values * 10 + rng.normal(0, 0.5, n),
        index=fwd_returns.index,
        name="test_signal",
    )

    base_eval = AlphaEvaluator(
        existing_features = pd.DataFrame(),
        spread_cost_bps   = 5.0,
    )
    result = base_eval.evaluate(signal, fwd_returns, regimes)
    print(result.summary())
    print(f"  decay_curve : {[f'{v:.4f}' for v in result.ic_decay_curve]}")
    print(f"  ic_by_regime: {result.ic_by_regime}")

    passed, failures = base_eval.passes_thresholds(result)
    print(f"  threshold_check: {'PASS ✓' if passed else 'FAIL ✗'}")
    if failures:
        for f in failures:
            print(f"    • {f}")
    print()


def test_multi_agent(fwd_returns: pd.DataFrame, regimes: pd.Series):
    _section("Test 3 — Multi-Agent Evaluator")

    rng    = np.random.default_rng(1)
    n      = len(fwd_returns)
    signal = pd.Series(
        -fwd_returns["ret_1h"].values * 8 + rng.normal(0, 0.5, n),
        index=fwd_returns.index,
        name="multiagent_test",
    )

    base      = AlphaEvaluator(pd.DataFrame())
    mae       = MultiAgentEvaluator(base)
    ma_result = mae.evaluate(signal, fwd_returns, regimes)

    print(ma_result.verdict_table())
    print(f"\n  overall_pass: {ma_result.overall_pass}")
    if ma_result.improvement_suggestions:
        print("  suggestions:")
        for s in ma_result.improvement_suggestions:
            print(f"    → {s}")
    print()


def test_knowledge_base():
    _section("Test 4 — Knowledge Base")

    import tempfile, shutil
    tmp  = Path(tempfile.mkdtemp())
    kb   = KnowledgeBase(tmp)

    # Check seeded files
    dsl  = kb.load_skill("midas-dsl")
    assert "ts_zscore" in dsl, "DSL skill not seeded"
    print(f"  ✓ DSL skill seeded ({len(dsl)} chars)")

    plan_prompt = kb.load_prompt("plan")
    assert "{{data_schema}}" in plan_prompt
    print("  ✓ Plan prompt seeded")

    # Save & load learning
    kb.save_offline_learning("test-pattern", "# Test Learning\n\nSome insight.")
    recent = kb.load_recent_learnings(n=5)
    assert "Test Learning" in recent
    print("  ✓ Offline learning saved and retrieved")

    # Save & load candidate
    kb.save_candidate("my_signal", "# Candidate: my_signal\n\n**status:** candidate")
    assert "my_signal" in kb.list_features("candidates")
    print("  ✓ Candidate feature saved")

    shutil.rmtree(tmp)
    print()


def test_feature_promoter():
    _section("Test 5 — Feature Promoter")

    import tempfile, shutil
    tmp      = Path(tempfile.mkdtemp())
    kb       = KnowledgeBase(tmp)
    promoter = FeaturePromoter(kb)

    # Plant a candidate
    kb.save_candidate("alpha_signal", "# Candidate: alpha_signal\n\n**status:** candidate  |  **discovered:** 2025-01-01")

    assert "alpha_signal" in promoter.list_candidates()
    print("  ✓ Candidate present")

    ok = promoter.promote("alpha_signal")
    assert ok
    assert "alpha_signal" in promoter.list_deployed()
    assert "alpha_signal" not in promoter.list_candidates()
    print("  ✓ Promoted to deployed")

    ok = promoter.demote("alpha_signal", reason="IC degraded after regime shift")
    assert ok
    assert "alpha_signal" in promoter.list_archived()
    assert "alpha_signal" not in promoter.list_deployed()
    print("  ✓ Demoted to archived")

    promoter.print_pipeline()
    shutil.rmtree(tmp)


def test_offline_loop(data_fn, fwd_returns, regimes):
    _section("Test 6 — Offline Compound Loop (mock LLM)")

    import tempfile, shutil
    tmp  = Path(tempfile.mkdtemp())
    kb   = KnowledgeBase(tmp)
    llm  = MockLLM()

    base      = AlphaEvaluator(pd.DataFrame())
    evaluator = MultiAgentEvaluator(base)
    proposer  = ExpressionProposer(kb, llm)
    cfg       = OfflineLoopConfig(max_iterations=4, verbose=True)
    loop      = OfflineCompoundLoop(kb, evaluator, proposer, cfg)

    learning = loop.run(
        research_goal    = "Find a mean-reversion signal based on VWAP deviation",
        existing_factors = [],
        data_fn          = data_fn,
        regime           = "HIGH_VOL",
    )

    print(f"\n  LLM calls made : {llm.call_count}")
    print(f"  result         : {learning.result}")
    print(f"  pattern        : {learning.pattern_identified}")
    print(f"  expression     : {learning.expression[:60]!r}")

    # Knowledge should be persisted
    recent = kb.load_recent_learnings(n=3)
    assert learning.pattern_identified in recent or len(recent) > 0
    print("  ✓ Learning persisted to knowledge base")

    shutil.rmtree(tmp)
    print()


def test_online_monitor(fwd_returns: pd.DataFrame):
    _section("Test 7 — Online Monitor (mock LLM)")

    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    kb  = KnowledgeBase(tmp)
    llm = MockLLM()

    killed: list  = []
    alerted: list = []

    monitor = OnlineMonitor(
        kb            = kb,
        feature_names = ["vwap_zscore", "rsi_inversion"],
        llm_client    = llm,
        on_alert      = lambda a: alerted.append(a),
        on_kill       = lambda f: killed.append(f),
        # Very low thresholds so we actually trigger an alert
        alert_thresholds={
            "ic_ratio_warning":        0.99,   # always trigger warning
            "ic_ratio_critical":       0.98,   # always trigger critical
            "slippage_ratio_warning":  9999.0,
            "slippage_ratio_critical": 9999.0,
            "pnl_drawdown_bps":       -9999.0,
        },
    )

    rng = np.random.default_rng(2)
    ts  = datetime(2025, 1, 1, 0, 0)

    # Push 30 bars so rolling buffers have data
    for i in range(30):
        ts += timedelta(hours=1)
        asyncio.run(monitor.process_update(
            timestamp      = ts,
            feature_values = {
                "vwap_zscore":  float(rng.normal(0, 1)),
                "rsi_inversion": float(rng.normal(0, 0.5)),
            },
            forward_return = float(rng.normal(0, 0.01)),
            regime         = "HIGH_VOL",
            market_context = {"btc_vol": "high", "regime": "HIGH_VOL"},
        ))

    print(f"  Bars processed : 30")
    print(f"  Alerts fired   : {len(alerted)}")
    print(f"  Kill signals   : {len(killed)}")
    if alerted:
        a = alerted[0]
        print(f"  Sample alert   : {a.feature_name} — {a.alert_type.value} ({a.severity})")

    report = monitor.generate_daily_report("2025-01-01")
    print(f"  Daily report   : saved ({len(report.feature_breakdown)} feature rows)")
    print(f"  LLM calls      : {llm.call_count}")

    shutil.rmtree(tmp)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  MIDAS — Integration Test Suite")
    print("█" * 60)

    # Shared synthetic data
    data_fn, fwd_returns, regimes = make_synthetic_data(n=500)

    tests = [
        ("DSL Validator",           lambda: test_dsl_validator()),
        ("Static Evaluator",        lambda: test_evaluator(fwd_returns, regimes)),
        ("Multi-Agent Evaluator",   lambda: test_multi_agent(fwd_returns, regimes)),
        ("Knowledge Base",          lambda: test_knowledge_base()),
        ("Feature Promoter",        lambda: test_feature_promoter()),
        ("Offline Compound Loop",   lambda: test_offline_loop(data_fn, fwd_returns, regimes)),
        ("Online Monitor",          lambda: test_online_monitor(fwd_returns)),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {name}")
            traceback.print_exc()
            failed += 1

    print("\n" + "─" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("─" * 60 + "\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
