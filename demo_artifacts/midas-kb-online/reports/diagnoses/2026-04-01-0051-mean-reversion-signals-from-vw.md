# Diagnosis Report — mean-reversion-signals-from-vwap-deviati

**Alert:** ic_decay (critical)
**Threshold:** ic_ratio=0.326 < 0.98
**Kill Signal:** True

## Root Cause

The alpha feature "mean-reversion-signals-from-vwap-deviati" is displaying a significant IC decay trend despite being in a high volatility regime, which historically supported its signal strength.
Traditionally, the 5-day ICs were significantly above the 30-day ICs (IC ratio > 2), indicating strong recent predictive power with some positive momentum.
However, over the last ~48 hours, the 5-day IC has rapidly declined from ~0.11 to near 0.012 while the 30-day IC has remained roughly steady around 0.035, resulting in an IC ratio dropping below 0.33.
This decay suggests the short-term predictive edge has almost vanished, indicating rapid loss of signal relevance at recent horizons.
The consistent high-volatility market regime with elevated BTC volatility implies environment alone is not driving degradation.
Also, PnL sharply turned negative concurrently with the IC decay, confirming alpha is no longer generating positive returns.
Hence, the root cause is structural drift or market microstructure change causing the mean-reversion signals on VWAP deviation to lose efficacy, rather than regime or volatility shifts.


## Evidence

  - IC_5d dropped consistently from ~0.11 to 0.012 over last 2 days while IC_30d remained ~0.035.
  - IC ratio dropped from above 2 (strong momentum) to below 0.33 (severe decay).
  - PnL shifted from positive hundreds to consistent negative hundreds concurrently with IC decay.
  - Market regime and BTC volatility stayed "HIGH_VOL", same as prior period where alpha worked well.
  - No past learnings or similar degradation observed previously.

## Proposed Fix

```
null

```

## Learning Document

# Online Learning: Diagnosing IC Decay in Mean-Reversion VWAP Deviation Signals

## Observation
The mean-reversion alpha signal derived from VWAP deviations sustained a sharp loss in short-term predictive power (IC_5d) while longer-term IC remained stable. This was accompanied by PnL deterioration and occurred despite an unchanged high-volatility market regime.

## Root Cause
The feature suffered from fast alpha decay likely caused by changes in market dynamics unrelated to volatility regime shifts. The structured mean-reversion relationship to VWAP deviations weakened, reducing signal quality.

## Pattern Identified
An initially strong recent IC compared to longer horizons, followed by rapid short-term IC decline and converging IC values, signals impending alpha degradation. Dissociation between IC and regime/volatility variables suggests the need for dynamic feature robustness checks.

## Action Taken
Immediate kill of the feature to prevent drawdown and alpha decay impact. Further investigation needed into feature construction and market microstructure shifts.

## Suggestions for Future
- Implement finer-grain diagnostics to catch early stage IC decay.
- Incorporate adaptive feature formulations that adjust with regime and microstructure changes.
- Monitor divergence patterns between IC_5d and IC_30d as a leading warning.
- Consider ensemble or multi-horizon signals to mitigate isolated horizon decay.

