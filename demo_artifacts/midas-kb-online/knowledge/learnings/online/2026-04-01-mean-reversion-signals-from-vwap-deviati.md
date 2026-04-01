# Online Learning: Diagnosing and Handling IC Decay in Mean Reversion Features

## Observation
The mean-reversion feature derived from VWAP deviations exhibited critical IC decay, with the short-term IC converging toward the long-term baseline and a corresponding drop in predictive quality as demonstrated by sharply negative PnL.

## Root Cause
The structural predictive relationship underpinning this feature has weakened, likely due to a shift in market dynamics within the same nominal regime, causing diminishing mean-reversion behavior exploitable by this metric.

## Pattern Identified
IC decay manifests first as a drop in short-term IC while long-term IC remains stable, accompanied by a ratio falling below alert thresholds. This decay often precedes persistent negative PnL and signals model feature degradation despite no regime re-classification.

## Action Taken
Immediate killing is not warranted due to stable regime identification and residual low but non-zero IC. A feature expression fix introducing volatility scaling and dynamic regime filtering is recommended to restore robustness.

## Suggestions for Future
- Implement volatility scaling and regime-aware recalibration in mean reversion features preemptively.
- Monitor IC ratio trends closely in conjunction with realized PnL for early detection.
- Incorporate adaptive feature expressions that reflect evolving market microstructure even within a stable regime.
