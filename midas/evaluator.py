"""
Midas Evaluator
===============
Static, deterministic, vectorised evaluation of alpha features.
No LLM calls in the inner loop — fast enough to run hundreds of
expressions per search session.

Two public classes:
  AlphaEvaluator      — computes all EvaluationResult fields
  MultiAgentEvaluator — runs six specialised agents in parallel,
                        returns MultiAgentResult with per-agent verdicts
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import (
    AgentVerdict,
    EvaluationResult,
    MultiAgentResult,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEvaluator — the static metric engine
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEvaluator:
    """
    Compute the full EvaluationResult for an alpha feature.

    Args:
        existing_features   : DataFrame of already-deployed factor values,
                              index = timestamp, columns = factor names.
                              Pass an empty DataFrame if none exist yet.
        spread_cost_bps     : One-way spread assumption in basis points.
        forward_periods     : List of forward horizons (in hours) for decay analysis.
        train_ratio         : Fraction of data used as in-sample.
        thresholds          : Pass/fail threshold dict (or None → defaults).
    """

    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "min_rank_ic":       0.02,
        "min_ir":            0.50,
        "max_turnover":      0.80,
        "max_correlation":   0.70,
        "max_overfit_ratio": 1.50,
        "min_oos_ic":        0.01,
        "min_composite":     0.30,
    }

    def __init__(
        self,
        existing_features:   pd.DataFrame,
        spread_cost_bps:     float        = 5.0,
        forward_periods:     List[int]    = None,
        train_ratio:         float        = 0.70,
        thresholds:          Optional[Dict[str, float]] = None,
    ):
        self.existing    = existing_features if not existing_features.empty else pd.DataFrame()
        self.spread_cost = spread_cost_bps / 10_000
        self.periods     = forward_periods or [1, 4, 8, 24, 48]
        self.train_ratio = train_ratio
        self.thresholds  = thresholds or dict(self.DEFAULT_THRESHOLDS)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        feature:         pd.Series,
        forward_returns: pd.DataFrame,
        regime_labels:   Optional[pd.Series] = None,
    ) -> EvaluationResult:
        """
        Run the complete evaluation suite.

        Args:
            feature         : Alpha signal (index = timestamp).
            forward_returns : Multi-horizon returns DataFrame;
                              columns should be ret_1h, ret_4h, … to match
                              forward_periods.  At minimum ret_1h is required.
            regime_labels   : Optional market-regime labels (same index).

        Returns:
            EvaluationResult dataclass with all metrics filled in.
        """
        df = self._align(feature, forward_returns, regime_labels)
        if len(df) < 50:
            # Not enough data — return zeros
            return EvaluationResult(feature_name=feature.name or "unknown")

        split = int(len(df) * self.train_ratio)
        train, test = df.iloc[:split], df.iloc[split:]

        feat = df["feature"]
        ret1 = df["ret_1h"]

        corrs  = self._correlations(feat)
        decay  = self._decay_curve(df)
        ic_reg = self._ic_by_regime(df)

        is_ic  = self._ic(train["feature"], train["ret_1h"])
        oos_ic = self._ic(test["feature"],  test["ret_1h"])
        overfit = is_ic / oos_ic if oos_ic > 0 else float("inf")

        rank_ic   = self._rank_ic(feat, ret1)
        ic_val    = self._ic(feat, ret1)
        ic_std    = self._ic_std(feat, ret1)
        ir_val    = ic_val / ic_std if ic_std > 0 else 0.0
        turnover  = self._turnover(feat)
        eff_ic    = ic_val - turnover * self.spread_cost * 252
        marg_ic   = self._marginal_ic(df)
        max_corr  = max(corrs.values(), default=0.0)
        q_spread  = self._quintile_spread(df)
        q_sharpe  = self._quintile_sharpe(df)
        hl        = self._half_life(decay)
        composite = self._composite(rank_ic, ir_val, eff_ic, marg_ic, oos_ic)

        return EvaluationResult(
            feature_name              = feature.name or "unknown",
            ic                        = ic_val,
            rank_ic                   = rank_ic,
            ic_std                    = ic_std,
            ir                        = ir_val,
            ic_decay_curve            = decay,
            half_life_hours           = hl,
            turnover                  = turnover,
            effective_ic              = eff_ic,
            correlation_with_existing = corrs,
            max_correlation           = max_corr,
            marginal_ic               = marg_ic,
            ic_by_regime              = ic_reg,
            ic_quintile_spread        = q_spread,
            sharpe_quintile           = q_sharpe,
            is_ic                     = is_ic,
            oos_ic                    = oos_ic,
            overfit_ratio             = overfit,
            composite_score           = composite,
        )

    def passes_thresholds(self, r: EvaluationResult) -> tuple[bool, List[str]]:
        """Return (passed, list_of_failure_messages)."""
        t = self.thresholds
        failures = []

        checks = [
            (r.rank_ic       < t["min_rank_ic"],       f"rank_ic {r.rank_ic:.4f} < {t['min_rank_ic']}"),
            (r.ir            < t["min_ir"],             f"ir {r.ir:.2f} < {t['min_ir']}"),
            (r.turnover      > t["max_turnover"],       f"turnover {r.turnover:.2f} > {t['max_turnover']}"),
            (r.max_correlation > t["max_correlation"],  f"max_correlation {r.max_correlation:.2f} > {t['max_correlation']}"),
            (r.overfit_ratio > t["max_overfit_ratio"],  f"overfit_ratio {r.overfit_ratio:.2f} > {t['max_overfit_ratio']}"),
            (r.oos_ic        < t["min_oos_ic"],         f"oos_ic {r.oos_ic:.4f} < {t['min_oos_ic']}"),
            (r.composite_score < t["min_composite"],    f"composite {r.composite_score:.2f} < {t['min_composite']}"),
        ]

        for failed, msg in checks:
            if failed:
                failures.append(msg)

        return (len(failures) == 0, failures)

    # ─────────────────────────────────────────────
    # Private metric helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _ic(feat: pd.Series, ret: pd.Series) -> float:
        v = feat.corr(ret)
        return float(v) if pd.notna(v) else 0.0

    @staticmethod
    def _rank_ic(feat: pd.Series, ret: pd.Series) -> float:
        v = feat.rank().corr(ret.rank())
        return float(v) if pd.notna(v) else 0.0

    @staticmethod
    def _ic_std(feat: pd.Series, ret: pd.Series, window: int = 20) -> float:
        rolling = feat.rolling(window).corr(ret)
        v = rolling.std()
        return float(v) if pd.notna(v) else 1e-9

    @staticmethod
    def _turnover(feat: pd.Series) -> float:
        pos = np.sign(feat) * np.minimum(np.abs(feat), 1)
        return float(pos.diff().abs().mean())

    def _decay_curve(self, df: pd.DataFrame) -> List[float]:
        curve = []
        for p in self.periods:
            col = f"ret_{p}h"
            if col in df.columns:
                curve.append(self._ic(df["feature"], df[col]))
            else:
                curve.append(float("nan"))
        return curve

    def _half_life(self, curve: List[float]) -> float:
        if not curve or np.isnan(curve[0]) or curve[0] <= 0:
            return float("nan")
        target = curve[0] * 0.5
        for i, ic in enumerate(curve):
            if pd.isna(ic):
                continue
            if ic <= target:
                if i == 0:
                    return float(self.periods[0])
                prev = curve[i - 1]
                if pd.isna(prev) or prev == ic:
                    return float(self.periods[i])
                frac = (prev - target) / (prev - ic)
                return float(self.periods[i - 1] + frac * (self.periods[i] - self.periods[i - 1]))
        return float(self.periods[-1])

    def _correlations(self, feat: pd.Series) -> Dict[str, float]:
        if self.existing.empty:
            return {}
        corrs: Dict[str, float] = {}
        for col in self.existing.columns:
            aligned = self.existing[col].reindex(feat.index)
            v = feat.corr(aligned)
            corrs[col] = float(v) if pd.notna(v) else 0.0
        return corrs

    def _marginal_ic(self, df: pd.DataFrame) -> float:
        """IC of the residual after regressing out existing factors."""
        if self.existing.empty:
            return self._ic(df["feature"], df["ret_1h"])
        try:
            from sklearn.linear_model import LinearRegression
            X = self.existing.reindex(df.index).dropna(axis=1)
            common = X.index.intersection(df.index)
            if len(common) < 100:
                return self._ic(df["feature"], df["ret_1h"])
            y = df.loc[common, "feature"]
            reg = LinearRegression().fit(X.loc[common], y)
            residual = y - pd.Series(reg.predict(X.loc[common]), index=common)
            return self._ic(residual, df.loc[common, "ret_1h"])
        except Exception:
            return self._ic(df["feature"], df["ret_1h"])

    def _ic_by_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        if "regime" not in df.columns:
            return {}
        result: Dict[str, float] = {}
        for regime in df["regime"].dropna().unique():
            subset = df[df["regime"] == regime]
            if len(subset) >= 50:
                result[str(regime)] = self._ic(subset["feature"], subset["ret_1h"])
        return result

    def _quintile_spread(self, df: pd.DataFrame) -> float:
        try:
            q = pd.qcut(df["feature"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            q5 = df.loc[q == 5, "ret_1h"].mean()
            q1 = df.loc[q == 1, "ret_1h"].mean()
            return float(q5 - q1) if pd.notna(q5) and pd.notna(q1) else 0.0
        except Exception:
            return 0.0

    def _quintile_sharpe(self, df: pd.DataFrame) -> float:
        try:
            q = pd.qcut(df["feature"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            n = min(len(df[q == 5]), len(df[q == 1]))
            if n == 0:
                return 0.0
            ls = (df.loc[q == 5, "ret_1h"].values[:n] -
                  df.loc[q == 1, "ret_1h"].values[:n])
            std = ls.std()
            if std == 0:
                return 0.0
            return float(ls.mean() / std * np.sqrt(252 * 24))
        except Exception:
            return 0.0

    @staticmethod
    def _composite(rank_ic: float, ir: float, eff_ic: float,
                   marg_ic: float, oos_ic: float) -> float:
        """
        Weighted composite score normalised to [0, 1].

        Weights:
          rank_ic  30%  — primary signal quality
          ir       20%  — risk-adjusted predictability
          eff_ic   20%  — after-cost quality
          marg_ic  15%  — diversification value
          oos_ic   15%  — robustness
        """
        def _clip(v: float, ref: float) -> float:
            return float(np.clip(v / ref, 0.0, 1.0)) if ref > 0 else 0.0

        return (
            0.30 * _clip(rank_ic, 0.10) +
            0.20 * _clip(ir,      2.00) +
            0.20 * _clip(eff_ic,  0.08) +
            0.15 * _clip(marg_ic, 0.05) +
            0.15 * _clip(oos_ic,  0.08)
        )

    @staticmethod
    def _align(
        feature:         pd.Series,
        forward_returns: pd.DataFrame,
        regimes:         Optional[pd.Series],
    ) -> pd.DataFrame:
        df = pd.DataFrame({"feature": feature})
        for col in forward_returns.columns:
            df[col] = forward_returns[col]
        if regimes is not None:
            df["regime"] = regimes
        return df.dropna(subset=["feature", "ret_1h"])


# ─────────────────────────────────────────────────────────────────────────────
# Specialised evaluation agents
# ─────────────────────────────────────────────────────────────────────────────

class _Agent:
    name: str = "base"

    def evaluate(self, metrics: EvaluationResult) -> AgentVerdict:
        raise NotImplementedError


class _PredictivePowerAgent(_Agent):
    name = "predictive_power"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        passed = m.rank_ic >= 0.02 and m.ir >= 0.50
        sug = []
        if m.rank_ic < 0.02:
            sug.append("IC too low — try a different transformation or signal combination")
        if m.ir < 0.50:
            sug.append("IC unstable — add EMA smoothing or longer lookback to stabilise")
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=float(np.clip(m.rank_ic / 0.05, 0, 1)),
            details=f"rankIC={m.rank_ic:.4f}  IR={m.ir:.2f}",
            suggestions=sug,
        )


class _DecayAgent(_Agent):
    name = "decay_analysis"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        hl = m.half_life_hours
        passed = pd.notna(hl) and hl >= 4.0
        sug = []
        if not passed:
            sug.append(
                f"Alpha half-life {hl:.1f}h is too short for reliable execution; "
                "increase lookback or use as an entry-only signal"
            )
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=float(np.clip(hl / 24, 0, 1)) if pd.notna(hl) else 0.0,
            details=f"half_life={hl:.1f}h  decay={[f'{v:.4f}' for v in m.ic_decay_curve]}",
            suggestions=sug,
        )


class _TradingCostAgent(_Agent):
    name = "trading_cost"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        passed = m.effective_ic > 0 and m.turnover <= 0.80
        sug = []
        if m.effective_ic <= 0:
            sug.append("Negative effective IC after costs — add ema() smoothing to cut turnover")
        if m.turnover > 0.80:
            sug.append(f"Turnover {m.turnover:.2f} too high — apply ema() or a threshold filter")
        raw = m.ic if m.ic != 0 else 1e-9
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=float(np.clip(m.effective_ic / raw, 0, 1)),
            details=f"turnover={m.turnover:.2f}  rawIC={m.ic:.4f}  effIC={m.effective_ic:.4f}",
            suggestions=sug,
        )


class _DiversificationAgent(_Agent):
    name = "diversification"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        passed = m.max_correlation <= 0.70 and m.marginal_ic > 0.01
        sug = []
        if m.max_correlation > 0.70:
            high = [k for k, v in m.correlation_with_existing.items() if v > 0.50]
            sug.append(f"High correlation with {high} — orthogonalise or combine")
        if m.marginal_ic <= 0.01:
            sug.append("No marginal IC — feature is redundant with existing factors")
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=float(np.clip(1 - m.max_correlation, 0, 1)),
            details=f"maxCorr={m.max_correlation:.2f}  marginalIC={m.marginal_ic:.4f}",
            suggestions=sug,
        )


class _OverfitAgent(_Agent):
    name = "overfit_detection"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        passed = m.overfit_ratio <= 1.50 and m.oos_ic > 0.01
        sug = []
        if m.overfit_ratio > 1.50:
            sug.append(f"Overfit ratio {m.overfit_ratio:.2f} — simplify expression or reduce free parameters")
        if m.oos_ic <= 0.01:
            sug.append("OOS IC near zero — feature may be spurious; test on more diverse data")
        score = float(np.clip(1 - (m.overfit_ratio - 1) / 2, 0, 1))
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=score,
            details=f"IS_IC={m.is_ic:.4f}  OOS_IC={m.oos_ic:.4f}  ratio={m.overfit_ratio:.2f}",
            suggestions=sug,
        )


class _RegimeRobustnessAgent(_Agent):
    name = "regime_robustness"

    def evaluate(self, m: EvaluationResult) -> AgentVerdict:
        if not m.ic_by_regime:
            return AgentVerdict(
                agent_name=self.name,
                passed=True,
                score=0.5,
                details="No regime data available",
                suggestions=["Provide regime labels for robustness analysis"],
            )
        ics   = list(m.ic_by_regime.values())
        mn    = min(ics)
        rng   = max(ics) - mn
        passed = mn > 0 and rng < 0.05
        sug = []
        if mn <= 0:
            bad = [k for k, v in m.ic_by_regime.items() if v <= 0]
            sug.append(f"Negative IC in regimes {bad} — add regime filter")
        if rng >= 0.05:
            sug.append(f"IC variance {rng:.3f} across regimes — feature is strongly regime-conditional")
        return AgentVerdict(
            agent_name=self.name,
            passed=passed,
            score=float(np.clip(mn / 0.03, 0, 1)),
            details=f"ic_by_regime={m.ic_by_regime}",
            suggestions=sug,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentEvaluator — orchestrates all agents in parallel
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEvaluator:
    """
    Runs six specialised evaluation agents in parallel threads.
    Synthesises per-agent verdicts into a single MultiAgentResult.
    """

    AGENTS = [
        _PredictivePowerAgent(),
        _DecayAgent(),
        _TradingCostAgent(),
        _DiversificationAgent(),
        _OverfitAgent(),
        _RegimeRobustnessAgent(),
    ]

    def __init__(self, base_evaluator: AlphaEvaluator):
        self.base = base_evaluator

    def evaluate(
        self,
        feature:         pd.Series,
        forward_returns: pd.DataFrame,
        regime_labels:   Optional[pd.Series] = None,
    ) -> MultiAgentResult:
        # 1) Compute base metrics deterministically
        metrics = self.base.evaluate(feature, forward_returns, regime_labels)

        # 2) Run all agents in parallel
        verdicts: List[AgentVerdict] = []
        with ThreadPoolExecutor(max_workers=len(self.AGENTS)) as ex:
            futures = {ex.submit(agent.evaluate, metrics): agent.name
                       for agent in self.AGENTS}
            for fut in as_completed(futures):
                try:
                    verdicts.append(fut.result())
                except Exception as e:
                    verdicts.append(AgentVerdict(
                        agent_name=futures[fut],
                        passed=False,
                        score=0.0,
                        details=f"Agent crashed: {e}",
                        suggestions=[],
                    ))

        # 3) Synthesise
        blocking    = [v.details for v in verdicts if not v.passed]
        suggestions = list({s for v in verdicts for s in v.suggestions})

        return MultiAgentResult(
            feature_name            = metrics.feature_name,
            metrics                 = metrics,
            verdicts                = verdicts,
            overall_pass            = all(v.passed for v in verdicts),
            blocking_issues         = blocking,
            improvement_suggestions = suggestions,
        )
