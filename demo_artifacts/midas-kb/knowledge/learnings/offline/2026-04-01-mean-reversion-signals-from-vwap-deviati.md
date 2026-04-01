# Offline Learning: Mean-reversion signals from VWAP deviation require stable, regime-agnostic parameterization to maintain predictive consistency.


**date:** 2026-04-01  |  **result:** failed

---

## Expression

```
ema(
  if_else(
    regime == "TRENDING",
    ts_zscore(div(sub(close, vwap), vwap), 96),
    ts_zscore(div(sub(close, vwap), vwap), 96)
  ),
  16
)

```

## Why It Failed

The tested mean-reversion signals based on VWAP deviation failed to improve predictive power across regimes because tuning the lookback periods and smoothing parameters did not enhance rank IC, and regime-based conditional logic offered no benefit over a simple baseline z-score calculation.


## Pattern Identified

Mean-reversion signals from VWAP deviation require stable, regime-agnostic parameterization to maintain predictive consistency.


## Suggestions for Future

- Explore additional transformations or normalization methods beyond z-scoring to capture non-linear mean-reversion effects.
- Investigate alternate regime classification schemes or incorporate volatility-adjusted dynamic lookbacks rather than fixed periods.

## Key Metrics

```json
{
  "feature_name": "refined_1issues",
  "ic": 0.486490342265966,
  "rank_ic": 0.4838451033804135,
  "ic_std": 0.15536501404546427,
  "ir": 3.131273441803345,
  "half_life_hours": 33.62001792509295,
  "turnover": 0.7712661480295699,
  "effective_ic": 0.3893108076142402,
  "max_correlation": 0.0,
  "marginal_ic": 0.486490342265966,
  "ic_quintile_spread": 0.0831391050658164,
  "sharpe_quintile": 100.33124273509621,
  "is_ic": 0.4601657705488615,
  "oos_ic": 0.5462419497120794,
  "overfit_ratio": 0.8424211483417045,
  "composite_score": 1.0
}
```

## Related Learnings

- Similar VWAP-based signals often suffer from regime sensitivity undermining robustness.
- Prior experiments show smoothing z-score inputs rarely improve IR without regime-specific tailoring.
