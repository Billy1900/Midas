# Midas Demo Report

## Run Summary

- Provider: `openai`
- Model: `gpt-4.1-mini`
- Offline result: `failed`
- Promoted first candidate: `False`
- Online feature used: `mean-reversion-signals-from-vwap-deviati`
- Online alerts: `7`
- Kill signals: `2`
- Mock/API call count: `None`

## What I Ran

1. Bootstrapped a fresh offline knowledge base at `demo_artifacts/midas-kb`.
2. Ran one offline discovery session on bundled synthetic data.
3. Bootstrapped a fresh online knowledge base at `demo_artifacts/midas-kb-online`.
4. Ran `777` online metric updates and generated a daily report.

## Offline Result

- Pattern identified: `Mean-reversion signals from VWAP deviation require stable, regime-agnostic parameterization to maintain predictive consistency.`
- Expression:

```text
ema(
  if_else(
    regime == "TRENDING",
    ts_zscore(div(sub(close, vwap), vwap), 96),
    ts_zscore(div(sub(close, vwap), vwap), 96)
  ),
  16
)
```

- Key metrics:

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

## Online Result

- Portfolio metrics:

```json
{
  "avg_ic_5d": 0.03766063898816749,
  "total_pnl_bps": 54101.73420188652,
  "avg_turnover": 0.805713566552502,
  "n_alerts": 7,
  "n_critical": 7
}
```

- First alerts:

```json
[
  {
    "feature_name": "mean-reversion-signals-from-vwap-deviati",
    "alert_type": "ic_decay",
    "severity": "critical",
    "threshold": "ic_ratio=0.922 < 0.98"
  },
  {
    "feature_name": "mean-reversion-signals-from-vwap-deviati",
    "alert_type": "ic_decay",
    "severity": "critical",
    "threshold": "ic_ratio=0.810 < 0.98"
  },
  {
    "feature_name": "mean-reversion-signals-from-vwap-deviati",
    "alert_type": "ic_decay",
    "severity": "critical",
    "threshold": "ic_ratio=0.843 < 0.98"
  },
  {
    "feature_name": "mean-reversion-signals-from-vwap-deviati",
    "alert_type": "ic_decay",
    "severity": "critical",
    "threshold": "ic_ratio=0.856 < 0.98"
  },
  {
    "feature_name": "mean-reversion-signals-from-vwap-deviati",
    "alert_type": "ic_decay",
    "severity": "critical",
    "threshold": "ic_ratio=0.370 < 0.98"
  }
]
```

## Evaluation

The framework now runs out of the box through a real Python package and CLI.
The offline and online loops both execute end to end, persist markdown outputs,
and can be driven either by the mock provider or a real LLM provider.

The research quality of this demo should not be over-interpreted because it
uses synthetic data. What it does prove is that the orchestration, persistence,
and evaluation flow are wired up correctly.
