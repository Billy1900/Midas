"""
Midas Online Monitor
====================
The live-trading arm of the dual-loop system.

Flow:
  Deploy → Monitor (rolling metrics) → Alert → Diagnose → Learn
                ↑_________________________________________________|

Four classes:
  MonitorEngine   — rolling metric computation (pure maths, no LLM)
  AlertEngine     — threshold-based alert evaluation
  DiagnoseAgent   — LLM-powered root-cause analysis + learning generation
  OnlineMonitor   — orchestrates all three; call process_update() each bar
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .kb     import KnowledgeBase
from .models import (
    Alert, AlertType, DailyReport, DiagnoseResult, FeatureMetrics,
)

logger = logging.getLogger("midas.online")


# ─────────────────────────────────────────────────────────────────────────────
# MonitorEngine — pure rolling-metric computation
# ─────────────────────────────────────────────────────────────────────────────

class MonitorEngine:
    """
    Maintains in-memory rolling buffers for each deployed feature and
    computes FeatureMetrics on every call to update().

    Designed to run in a tight loop:
        for bar in live_feed:
            metrics = engine.update(bar.ts, bar.features, bar.return1h, ...)
    """

    _WINDOWS_HOURS = {"1d": 24, "5d": 120, "30d": 720}

    def __init__(self, feature_names: List[str]):
        self.feature_names  = feature_names
        self._feat: Dict[str, pd.DataFrame] = {}   # name → DataFrame(value, regime)
        self._ret:  pd.Series               = pd.Series(dtype=float)
        self._fill: pd.DataFrame            = pd.DataFrame()
        self._max_hours                     = self._WINDOWS_HOURS["30d"]

    def update(
        self,
        timestamp:      datetime,
        feature_values: Dict[str, float],
        forward_return: float,
        fills:          Optional[Dict[str, Any]] = None,
        regime:         str                       = "UNKNOWN",
    ) -> List[FeatureMetrics]:
        """
        Ingest one new bar and return the latest metrics for every feature.

        Args:
            timestamp      : Bar close time.
            feature_values : {feature_name: signal_value} for this bar.
            forward_return : 1-hour forward return (fraction, not bps).
            fills          : Optional execution stats dict.
            regime         : Market regime label.

        Returns:
            List of FeatureMetrics (one per feature_name in feature_values).
        """
        self._append(timestamp, feature_values, forward_return, fills, regime)
        return [
            m for name in self.feature_names
            if name in feature_values
            for m in [self._compute(name, timestamp, regime)]
            if m is not None
        ]

    # ─── internal helpers ────────────────────────────────────────────────────

    def _append(self, ts, values, ret, fills, regime):
        # Feature buffers
        for name, val in values.items():
            if name not in self._feat:
                self._feat[name] = pd.DataFrame(
                    columns=["value", "regime"], dtype=object
                )
            self._feat[name].loc[ts] = {"value": float(val), "regime": regime}

        # Return buffer
        self._ret.loc[ts] = float(ret)

        # Fills
        if fills:
            for k, v in fills.items():
                self._fill.loc[ts, k] = v

        # Trim to 30-day rolling window
        cutoff = ts - timedelta(hours=self._max_hours)
        for name in list(self._feat):
            self._feat[name] = self._feat[name].loc[self._feat[name].index > cutoff]
        self._ret  = self._ret[self._ret.index > cutoff]
        self._fill = self._fill[self._fill.index > cutoff] if not self._fill.empty else self._fill

    def _compute(self, name: str, ts: datetime, regime: str) -> Optional[FeatureMetrics]:
        buf = self._feat.get(name)
        if buf is None or len(buf) < self._WINDOWS_HOURS["1d"]:
            return None

        common = buf.index.intersection(self._ret.index)
        if len(common) < 24:
            return None

        feat = buf.loc[common, "value"].astype(float)
        ret  = self._ret.loc[common].astype(float)

        def rolling_ic(n: int) -> float:
            if len(feat) < n:
                return float("nan")
            f, r = feat.tail(n), ret.tail(n)
            v = f.corr(r)
            return float(v) if pd.notna(v) else float("nan")

        ic_1d  = rolling_ic(self._WINDOWS_HOURS["1d"])
        ic_5d  = rolling_ic(self._WINDOWS_HOURS["5d"])
        ic_30d = rolling_ic(self._WINDOWS_HOURS["30d"])
        ratio  = (ic_5d / ic_30d) if (pd.notna(ic_30d) and ic_30d != 0) else float("nan")

        # Turnover
        pos      = np.sign(feat) * np.minimum(np.abs(feat), 1.0)
        turnover = float(pos.diff().abs().tail(self._WINDOWS_HOURS["1d"]).mean())

        # Slippage
        slippage = 0.0
        if "slippage_bps" in self._fill.columns and not self._fill.empty:
            recent = self._fill["slippage_bps"].tail(self._WINDOWS_HOURS["1d"])
            slippage = float(recent.mean()) if len(recent) else 0.0

        # PnL attribution (simplified)
        pnl = float((feat * ret).tail(self._WINDOWS_HOURS["1d"]).sum() * 10_000)

        # IC in current regime
        regime_mask = buf.loc[common, "regime"] == regime
        ic_regime   = float("nan")
        if regime_mask.sum() >= 24:
            v = feat[regime_mask].corr(ret[regime_mask])
            ic_regime = float(v) if pd.notna(v) else float("nan")

        hl = self._estimate_half_life(feat, ret)

        return FeatureMetrics(
            feature_name          = name,
            timestamp             = ts,
            ic_1d                 = ic_1d,
            ic_5d                 = ic_5d,
            ic_30d                = ic_30d,
            ic_ratio              = ratio,
            half_life_hours       = hl,
            realized_turnover     = turnover,
            slippage_vs_expected  = slippage,
            pnl_contribution_bps  = pnl,
            current_regime        = regime,
            ic_in_regime          = ic_regime,
        )

    @staticmethod
    def _estimate_half_life(feat: pd.Series, ret: pd.Series) -> float:
        lags  = [1, 4, 8, 24, 48]
        ics: List[float] = []
        for lag in lags:
            lagged = ret.shift(-lag).dropna()
            common = feat.index.intersection(lagged.index)
            if len(common) < 24:
                ics.append(float("nan"))
                continue
            v = feat.loc[common].corr(lagged.loc[common])
            ics.append(float(v) if pd.notna(v) else float("nan"))

        if not ics or np.isnan(ics[0]) or ics[0] <= 0:
            return float("nan")
        target = ics[0] * 0.5
        for i, ic in enumerate(ics):
            if np.isnan(ic):
                continue
            if ic <= target:
                if i == 0:
                    return float(lags[0])
                prev = ics[i - 1]
                if np.isnan(prev) or prev == ic:
                    return float(lags[i])
                frac = (prev - target) / (prev - ic)
                return float(lags[i - 1] + frac * (lags[i] - lags[i - 1]))
        return float(lags[-1])




# ─────────────────────────────────────────────────────────────────────────────
# AlertEngine
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLDS = {
    "ic_ratio_warning":        0.70,
    "ic_ratio_critical":       0.50,
    "slippage_ratio_warning":  1.50,
    "slippage_ratio_critical": 2.00,
    "pnl_drawdown_bps":       -50.0,   # running 1-day pnl worse than this → warning
}


class AlertEngine:
    """
    Evaluates a list of FeatureMetrics against configurable thresholds.
    Returns any Alert objects that were triggered.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)
        self.history:   List[Alert] = []

    def evaluate(self, metrics: List[FeatureMetrics]) -> List[Alert]:
        alerts: List[Alert] = []
        t = self.thresholds

        for m in metrics:
            # IC ratio alerts
            if pd.notna(m.ic_ratio):
                if m.ic_ratio < t["ic_ratio_critical"]:
                    alerts.append(Alert(
                        alert_type         = AlertType.IC_DECAY,
                        feature_name       = m.feature_name,
                        timestamp          = m.timestamp,
                        severity           = "critical",
                        details            = {"ic_5d": m.ic_5d, "ic_30d": m.ic_30d,
                                              "ic_ratio": m.ic_ratio},
                        threshold_breached = (
                            f"ic_ratio={m.ic_ratio:.3f} < {t['ic_ratio_critical']}"
                        ),
                    ))
                elif m.ic_ratio < t["ic_ratio_warning"]:
                    alerts.append(Alert(
                        alert_type         = AlertType.IC_DECAY,
                        feature_name       = m.feature_name,
                        timestamp          = m.timestamp,
                        severity           = "warning",
                        details            = {"ic_5d": m.ic_5d, "ic_30d": m.ic_30d,
                                              "ic_ratio": m.ic_ratio},
                        threshold_breached = (
                            f"ic_ratio={m.ic_ratio:.3f} < {t['ic_ratio_warning']}"
                        ),
                    ))

            # Slippage alerts
            if m.slippage_vs_expected > t["slippage_ratio_critical"]:
                alerts.append(Alert(
                    alert_type         = AlertType.EXECUTION_SLIPPAGE,
                    feature_name       = m.feature_name,
                    timestamp          = m.timestamp,
                    severity           = "critical",
                    details            = {"slippage_ratio": m.slippage_vs_expected},
                    threshold_breached = (
                        f"slippage={m.slippage_vs_expected:.2f}x > {t['slippage_ratio_critical']}x"
                    ),
                ))
            elif m.slippage_vs_expected > t["slippage_ratio_warning"]:
                alerts.append(Alert(
                    alert_type         = AlertType.EXECUTION_SLIPPAGE,
                    feature_name       = m.feature_name,
                    timestamp          = m.timestamp,
                    severity           = "warning",
                    details            = {"slippage_ratio": m.slippage_vs_expected},
                    threshold_breached = (
                        f"slippage={m.slippage_vs_expected:.2f}x > {t['slippage_ratio_warning']}x"
                    ),
                ))

            # PnL drawdown
            if m.pnl_contribution_bps < t["pnl_drawdown_bps"]:
                alerts.append(Alert(
                    alert_type         = AlertType.DRAWDOWN,
                    feature_name       = m.feature_name,
                    timestamp          = m.timestamp,
                    severity           = "warning",
                    details            = {"pnl_1d_bps": m.pnl_contribution_bps},
                    threshold_breached = (
                        f"pnl={m.pnl_contribution_bps:.1f}bps < {t['pnl_drawdown_bps']}bps"
                    ),
                ))

        self.history.extend(alerts)
        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# DiagnoseAgent — LLM-powered root-cause analysis
# ─────────────────────────────────────────────────────────────────────────────

class DiagnoseAgent:
    """
    Calls the LLM with alert context and recent feature history to produce:
      - Root-cause explanation
      - A proposed expression fix
      - A kill signal if warranted
      - A structured learning document
    """

    def __init__(
        self,
        kb:         KnowledgeBase,
        llm_client,
        model:      str = "claude-sonnet-4-20250514",
        max_tokens: int = 2_000,
    ):
        self.kb         = kb
        self.llm        = llm_client
        self.model      = model
        self.max_tokens = max_tokens

    def diagnose(
        self,
        alert:          Alert,
        feature_metrics: List[FeatureMetrics],
        market_context: Dict[str, Any],
    ) -> DiagnoseResult:
        """Synchronous diagnosis (wraps async internally)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._diagnose_async(alert, feature_metrics, market_context),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._diagnose_async(alert, feature_metrics, market_context)
                )
        except Exception:
            return asyncio.run(self._diagnose_async(alert, feature_metrics, market_context))

    async def _diagnose_async(
        self,
        alert:          Alert,
        feature_metrics: List[FeatureMetrics],
        market_context: Dict[str, Any],
    ) -> DiagnoseResult:
        past = self.kb.load_learnings_for_feature(alert.feature_name, n=5)

        history_rows = [
            {
                "ts":      m.timestamp.isoformat(),
                "ic_5d":  m.ic_5d,
                "ic_30d": m.ic_30d,
                "ratio":  m.ic_ratio,
                "regime": m.current_regime,
                "pnl":    m.pnl_contribution_bps,
            }
            for m in feature_metrics[-48:]   # last 48 bars
        ]

        prompt = f"""You are a senior quant researcher diagnosing alpha feature degradation.

## Alert
- Feature   : {alert.feature_name}
- Type      : {alert.alert_type.value}
- Severity  : {alert.severity}
- Triggered : {alert.threshold_breached}
- Details   : {json.dumps(alert.details)}

## Rolling Metric History (last 48 bars)
{json.dumps(history_rows, indent=2)}

## Market Context
{json.dumps(market_context, indent=2)}

## Related Past Learnings
{chr(10).join(past) or "None yet."}

## Task
1. Identify the root cause of degradation with evidence.
2. Propose a concrete expression fix (or null if none possible).
3. Decide if the feature should be killed (stopped immediately).
4. Write a concise learning document for future reference.

Output ONLY valid YAML inside a ```yaml block:

```yaml
root_cause: |
  Technical explanation

evidence:
  - observed data point 1
  - observed data point 2

fix_proposal: |
  Modified DSL expression (or null)

kill_signal: false

learning_doc: |
  # Online Learning: [title]

  ## Observation
  [What happened]

  ## Root Cause
  [Why]

  ## Pattern Identified
  [Generalizable insight]

  ## Action Taken
  [What to do]

  ## Suggestions for Future
  - item 1
  - item 2
```
"""

        response = self.llm.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )

        parsed = _parse_yaml_text(response.content[0].text)

        return DiagnoseResult(
            feature_name = alert.feature_name,
            alert        = alert,
            root_cause   = parsed.get("root_cause", ""),
            evidence     = parsed.get("evidence", []),
            fix_proposal = parsed.get("fix_proposal"),
            kill_signal  = bool(parsed.get("kill_signal", False)),
            learning_doc = parsed.get("learning_doc", ""),
        )

    def save_learning(self, diagnosis: DiagnoseResult):
        slug = f"{diagnosis.feature_name}-{diagnosis.alert.alert_type.value}"
        self.kb.save_online_learning(slug, diagnosis.learning_doc)
        logger.info(f"Online learning saved: {slug}")


# ─────────────────────────────────────────────────────────────────────────────
# OnlineMonitor — top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class OnlineMonitor:
    """
    Top-level controller for the online compound loop.

    Typical usage (async):
        monitor = OnlineMonitor(kb, feature_names, llm_client)
        async for bar in live_feed:
            await monitor.process_update(...)
        report = monitor.generate_daily_report("2025-01-15")

    Typical usage (sync):
        monitor = OnlineMonitor(kb, feature_names, llm_client)
        for bar in live_feed:
            asyncio.run(monitor.process_update(...))
    """

    def __init__(
        self,
        kb:            KnowledgeBase,
        feature_names: List[str],
        llm_client,
        model: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        on_alert:         Optional[Callable[[Alert], None]]   = None,
        on_kill:          Optional[Callable[[str],   None]]   = None,
    ):
        self.kb       = kb
        self.engine   = MonitorEngine(feature_names)
        self.alerter  = AlertEngine(alert_thresholds)
        self.diagnoser= DiagnoseAgent(kb, llm_client, model=model or "claude-sonnet-4-20250514")

        self.on_alert = on_alert or (lambda a: logger.warning(f"ALERT {a.severity}: {a.feature_name} — {a.threshold_breached}"))
        self.on_kill  = on_kill  or (lambda f: logger.critical(f"KILL SIGNAL: {f}"))

        # Accumulators (reset daily)
        self._daily_metrics: List[FeatureMetrics] = []
        self._daily_alerts:  List[Alert]           = []
        self._daily_diagnoses: List[DiagnoseResult] = []

        # History buffer for diagnose agent
        self._metric_history: Dict[str, List[FeatureMetrics]] = {
            name: [] for name in feature_names
        }

    async def process_update(
        self,
        timestamp:      datetime,
        feature_values: Dict[str, float],
        forward_return: float,
        fills:          Optional[Dict[str, Any]] = None,
        regime:         str                       = "UNKNOWN",
        market_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Process one bar update (call hourly or on each completed bar).

        This is the main compound-learning hook: critical alerts trigger
        async LLM diagnosis which writes a learning document automatically.
        """
        # 1) Compute rolling metrics
        metrics = self.engine.update(
            timestamp, feature_values, forward_return, fills, regime
        )
        self._daily_metrics.extend(metrics)

        # Update history buffers
        for m in metrics:
            self._metric_history.setdefault(m.feature_name, []).append(m)

        # 2) Evaluate alerts
        alerts = self.alerter.evaluate(metrics)
        for a in alerts:
            self.on_alert(a)
            self._daily_alerts.append(a)

            # 3) Critical → async diagnosis + learning
            if a.severity == "critical":
                history = self._metric_history.get(a.feature_name, [])
                diagnosis = await self.diagnoser._diagnose_async(
                    alert           = a,
                    feature_metrics = history,
                    market_context  = market_context or {},
                )
                self._daily_diagnoses.append(diagnosis)
                self.diagnoser.save_learning(diagnosis)   # COMPOUND!

                if diagnosis.kill_signal:
                    self.on_kill(a.feature_name)

                # Save diagnosis report
                report_txt = _format_diagnosis(diagnosis)
                self.kb.save_diagnosis_report(
                    f"{a.feature_name}-{a.alert_type.value}", report_txt
                )

    def generate_daily_report(self, date: str) -> DailyReport:
        """Generate and persist the end-of-day summary."""
        portfolio = self._aggregate_portfolio()
        regime_info = {}
        exec_info   = {}

        if self._daily_metrics:
            regime_info = {"current": self._daily_metrics[-1].current_regime}
            slippages   = [m.slippage_vs_expected for m in self._daily_metrics]
            exec_info   = {"avg_slippage_ratio": float(np.nanmean(slippages))}

        report = DailyReport(
            date              = date,
            portfolio_metrics = portfolio,
            feature_breakdown = list(self._daily_metrics),
            alerts_triggered  = list(self._daily_alerts),
            regime_info       = regime_info,
            execution_summary = exec_info,
        )

        self.kb.save_daily_report(date, _format_daily_report(report))
        logger.info(f"Daily report saved for {date}")

        # Reset accumulators
        self._daily_metrics   = []
        self._daily_alerts    = []
        self._daily_diagnoses = []

        return report

    def _aggregate_portfolio(self) -> Dict[str, Any]:
        if not self._daily_metrics:
            return {}
        ic5s     = [m.ic_5d for m in self._daily_metrics if pd.notna(m.ic_5d)]
        pnls     = [m.pnl_contribution_bps for m in self._daily_metrics]
        turnovers= [m.realized_turnover for m in self._daily_metrics]
        return {
            "avg_ic_5d":   float(np.nanmean(ic5s))     if ic5s     else float("nan"),
            "total_pnl_bps": float(np.sum(pnls))       if pnls     else 0.0,
            "avg_turnover":  float(np.nanmean(turnovers)) if turnovers else float("nan"),
            "n_alerts":    len(self._daily_alerts),
            "n_critical":  sum(1 for a in self._daily_alerts if a.severity == "critical"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_daily_report(r: DailyReport) -> str:
    pm = r.portfolio_metrics
    header = f"""# Alpha Performance Report — {r.date}

## Portfolio Summary

| metric         | value |
|----------------|-------|
| avg IC (5d)    | {pm.get('avg_ic_5d', float('nan')):.4f} |
| total PnL      | {pm.get('total_pnl_bps', 0):.1f} bps |
| avg turnover   | {pm.get('avg_turnover', float('nan')):.2f} |
| alerts         | {pm.get('n_alerts', 0)} ({pm.get('n_critical', 0)} critical) |

## Feature Breakdown

| feature | ic_5d | ic_30d | ratio | turnover | pnl (bps) | regime |
|---------|-------|--------|-------|----------|-----------|--------|
"""
    rows = []
    for m in r.feature_breakdown:
        rows.append(
            f"| {m.feature_name} "
            f"| {m.ic_5d:.4f} "
            f"| {m.ic_30d:.4f} "
            f"| {m.ic_ratio:.2f} "
            f"| {m.realized_turnover:.2f} "
            f"| {m.pnl_contribution_bps:.1f} "
            f"| {m.current_regime} |"
        )

    alert_section = "\n## Alerts\n\n"
    for a in r.alerts_triggered:
        alert_section += f"- **{a.severity.upper()}** [{a.feature_name}] {a.threshold_breached}\n"

    return header + "\n".join(rows) + alert_section + f"\n## Regime\n\nCurrent: {r.regime_info.get('current', 'UNKNOWN')}\n"


def _format_diagnosis(d: DiagnoseResult) -> str:
    evidence = "\n".join(f"  - {e}" for e in d.evidence)
    return f"""# Diagnosis Report — {d.feature_name}

**Alert:** {d.alert.alert_type.value} ({d.alert.severity})
**Threshold:** {d.alert.threshold_breached}
**Kill Signal:** {d.kill_signal}

## Root Cause

{d.root_cause}

## Evidence

{evidence}

## Proposed Fix

```
{d.fix_proposal or "None — consider retiring this feature"}
```

## Learning Document

{d.learning_doc}
"""


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
