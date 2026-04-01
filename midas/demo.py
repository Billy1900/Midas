from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd

from .evaluator import AlphaEvaluator, MultiAgentEvaluator
from .kb import KnowledgeBase, _slugify
from .llm import create_llm_client
from .loops import OfflineCompoundLoop, OfflineLoopConfig
from .monitor import OnlineMonitor
from .promoter import FeaturePromoter
from .proposer import ExpressionProposer


class MockMessage:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(text=text)]


class MockLLM:
    """Returns plausible YAML payloads for each prompt type."""

    def __init__(self):
        self.call_count = 0
        self.messages = self

    def create(self, **kwargs) -> MockMessage:
        self.call_count += 1
        content = kwargs.get("messages", [{}])[-1].get("content", "")
        lowered = content.lower()

        if "kill_signal" in lowered or "root_cause" in lowered:
            return MockMessage("""```yaml
root_cause: |
  The IC ratio has dropped because the feature relies on VWAP deviation
  which became less predictive during a low-volatility grinding regime.

evidence:
  - ic_5d weakened meaningfully versus ic_30d
  - turnover remained elevated while predictive power fell

fix_proposal: |
  ema(ts_zscore(div(sub(close, vwap), atr(high, low, close, 48)), 96), 6)

kill_signal: false

learning_doc: |
  # Online Learning: VWAP mean-reversion degrades in low-vol trending regime

  ## Observation
  IC_5d / IC_30d ratio dropped below the critical threshold.

  ## Root Cause
  Low-vol trending regimes suppress short-horizon mean-reversion.

  ## Pattern Identified
  Add a regime filter and lengthen the smoothing window.

  ## Action Taken
  Proposed a slower ATR-normalised VWAP deviation expression.

  ## Suggestions for Future
  - Gate mean-reversion signals when ATR percentile is too low
  - Compare regime-specific IC before keeping a live signal on
```""")

        if "iteration history" in lowered:
            return MockMessage("""```yaml
why_it_worked_or_failed: |
  The proposed signal was economically plausible but pointed in the wrong
  direction on the synthetic dataset, so predictive power stayed negative.

pattern_identified: |
  ATR-normalised VWAP deviation needs sign validation before promotion

suggestions_for_future:
  - Test the inverse sign of the same base signal
  - Add explicit directional sanity checks before refinement

related_learnings:
  - Synthetically generated alpha can still reject intuitive mean-reversion ideas
```""")

        if "blocking issues" in lowered:
            return MockMessage("""```yaml
refined_expression: |
  ts_zscore(ema(div(sub(close, vwap), atr(high, low, close, 24)), 4), 48)

changes_made:
  - "Wrapped the raw signal in ema(4) to reduce turnover"

expected_improvement:
  - "turnover should improve"
```""")

        if "previously fail" in lowered or "alpha expression generation" in lowered:
            return MockMessage("""```yaml
candidates:
  - name: "vwap_zscore_mean_rev"
    expression: |
      ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)
    rationale: "ATR-normalised VWAP deviation captures over-extension"

  - name: "rsi_inversion"
    expression: |
      sub(50, rsi(close, 14))
    rationale: "Classic short-horizon reversal idea"
```""")

        return MockMessage("""```yaml
hypothesis: |
  When close diverges from VWAP on an ATR-normalised basis, the move tends
  to mean-revert over the next hour.

target_horizon: "1h"

data_sources:
  - field: close
    rationale: mark price
  - field: vwap
    rationale: fair-value anchor
  - field: atr
    rationale: volatility normaliser

expression_sketch: |
  ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)

risks:
  - signal may flip direction in trending regimes

related_learnings: []
```""")


def make_synthetic_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    true_signal = pd.Series(rng.normal(0, 1, n), index=idx, name="true_signal")
    noise = rng.normal(0, 5, n)
    fwd_returns = pd.DataFrame(
        {
            "ret_1h": true_signal * 0.03 + noise * 0.01,
            "ret_4h": true_signal * 0.025 + rng.normal(0, 1, n) * 0.015,
            "ret_8h": true_signal * 0.020 + rng.normal(0, 1, n) * 0.020,
            "ret_24h": true_signal * 0.010 + rng.normal(0, 1, n) * 0.030,
            "ret_48h": true_signal * 0.005 + rng.normal(0, 1, n) * 0.040,
        },
        index=idx,
    )
    regimes = pd.Series(rng.choice(["HIGH_VOL", "LOW_VOL", "TRENDING"], n), index=idx, name="regime")

    def compute(expression: str) -> pd.Series:
        expr = expression.lower()
        series = true_signal.copy()
        if "ema(" in expr:
            series = series.ewm(span=4, adjust=False).mean()
        if "sub(50, rsi" in expr:
            series = -true_signal
        if "close, vwap" in expr:
            series = true_signal
        return pd.Series(series.values, index=idx)

    return compute, fwd_returns, regimes


def _create_demo_llm(provider: str, api_key: Optional[str], model: Optional[str]):
    if provider == "mock":
        return MockLLM(), "mock"
    llm, cfg = create_llm_client(api_key=api_key, provider=provider, model=model)
    return llm, cfg.model


def run_demo(
    base_dir: Path,
    provider: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    online_bars: int = 800,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    offline_kb = base_dir / "midas-kb"
    online_kb = base_dir / "midas-kb-online"
    for folder in (offline_kb, online_kb):
        if folder.exists():
            shutil.rmtree(folder)

    llm, chosen_model = _create_demo_llm(provider, api_key, model)
    compute_fn, fwd_returns, regimes = make_synthetic_data()

    kb = KnowledgeBase(offline_kb)
    evaluator = MultiAgentEvaluator(AlphaEvaluator(pd.DataFrame(), spread_cost_bps=5.0))
    proposer = ExpressionProposer(kb, llm, model=chosen_model)
    offline = OfflineCompoundLoop(kb, evaluator, proposer, OfflineLoopConfig(max_iterations=4, verbose=False, model=chosen_model))
    promoter = FeaturePromoter(kb)
    learning = offline.run(
        research_goal="Find a mean-reversion signal based on VWAP deviation",
        existing_factors=[],
        data_fn=lambda: (compute_fn, fwd_returns, regimes),
        regime="HIGH_VOL",
    )

    candidate_names = promoter.list_candidates()
    promoted = False
    if candidate_names:
        promoted = promoter.promote(candidate_names[0])

    online_feature_name = candidate_names[0] if candidate_names else _slugify(learning.pattern_identified)[:40] or "demo_feature"
    online_alerts: list[dict[str, Any]] = []
    kill_signals: list[str] = []

    online_kb_obj = KnowledgeBase(online_kb)
    monitor = OnlineMonitor(
        kb=online_kb_obj,
        feature_names=[online_feature_name],
        llm_client=llm,
        model=chosen_model,
        on_alert=lambda a: online_alerts.append(
            {
                "feature_name": a.feature_name,
                "alert_type": a.alert_type.value,
                "severity": a.severity,
                "threshold": a.threshold_breached,
            }
        ),
        on_kill=lambda name: kill_signals.append(name),
        alert_thresholds={
            "ic_ratio_warning": 0.99,
            "ic_ratio_critical": 0.98,
            "slippage_ratio_warning": 9999.0,
            "slippage_ratio_critical": 9999.0,
            "pnl_drawdown_bps": -9999.0,
        },
    )

    rng = np.random.default_rng(7)
    ts = datetime(2025, 1, 1, 0, 0)

    async def _process_all_updates() -> None:
        ts_local = ts
        for _ in range(online_bars):
            ts_local += timedelta(hours=1)
            await monitor.process_update(
                timestamp=ts_local,
                feature_values={online_feature_name: float(rng.normal(0, 1))},
                forward_return=float(rng.normal(0, 0.01)),
                regime="HIGH_VOL",
                market_context={"btc_vol": "high", "regime": "HIGH_VOL"},
            )

    asyncio.run(_process_all_updates())
    daily_report = monitor.generate_daily_report("2025-01-31")
    summary = {
        "provider": provider,
        "model": chosen_model,
        "offline_learning": learning,
        "candidate_names": candidate_names,
        "promoted_first_candidate": promoted,
        "online_feature_name": online_feature_name,
        "online_alerts": online_alerts,
        "kill_signals": kill_signals,
        "daily_report": daily_report,
        "offline_kb": str(offline_kb),
        "online_kb": str(online_kb),
        "llm_call_count": getattr(llm, "call_count", None),
    }

    final_report_path = report_path or base_dir / "DEMO_REPORT.md"
    _write_demo_report(final_report_path, summary)
    summary["report_path"] = str(final_report_path)
    return summary


def _write_demo_report(report_path: Path, summary: dict[str, Any]):
    learning = summary["offline_learning"]
    daily_report = summary["daily_report"]
    online_alerts = summary["online_alerts"]

    report = f"""# Midas Demo Report

## Run Summary

- Provider: `{summary['provider']}`
- Model: `{summary['model']}`
- Offline result: `{learning.result}`
- Promoted first candidate: `{summary['promoted_first_candidate']}`
- Online feature used: `{summary['online_feature_name']}`
- Online alerts: `{len(online_alerts)}`
- Kill signals: `{len(summary['kill_signals'])}`
- Mock/API call count: `{summary['llm_call_count']}`

## What I Ran

1. Bootstrapped a fresh offline knowledge base at `{summary['offline_kb']}`.
2. Ran one offline discovery session on bundled synthetic data.
3. Bootstrapped a fresh online knowledge base at `{summary['online_kb']}`.
4. Ran `{len(daily_report.feature_breakdown)}` online metric updates and generated a daily report.

## Offline Result

- Pattern identified: `{learning.pattern_identified.strip()}`
- Expression:

```text
{learning.expression.strip()}
```

- Key metrics:

```json
{json.dumps(learning.metrics, indent=2, default=str)}
```

## Online Result

- Portfolio metrics:

```json
{json.dumps(daily_report.portfolio_metrics, indent=2, default=str)}
```

- First alerts:

```json
{json.dumps(online_alerts[:5], indent=2)}
```

## Evaluation

The framework now runs out of the box through a real Python package and CLI.
The offline and online loops both execute end to end, persist markdown outputs,
and can be driven either by the mock provider or a real LLM provider.

The research quality of this demo should not be over-interpreted because it
uses synthetic data. What it does prove is that the orchestration, persistence,
and evaluation flow are wired up correctly.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
