# Diagnosis Report — mean-reversion-signals-from-vwap-deviati

**Alert:** ic_decay (critical)
**Threshold:** ic_ratio=0.859 < 0.98
**Kill Signal:** False

## Root Cause

The feature "mean-reversion-signals-from-vwap-deviati" shows significant IC decay, as evidenced by the drop in the 5-day IC (ic_5d) to near or below the 30-day IC (ic_30d) and the ic_ratio dropping below the alert threshold of 0.98, currently at 0.859. Historically, this feature had much higher IC values (ic_5d around 0.10-0.11 and ic_ratio consistently above 1.5 up to 2.4), but recently, the ic_5d sharply declined to as low as ~0.01-0.03 while the ic_30d remained relatively stable (~0.036). This flattening and decline in short-term IC indicates the feature's predictive power for mean reversion from VWAP deviations has faded in the current high volatility regime, potentially due to structural market changes or regime shifts that erode previously strong mean-reversion patterns.

The consistent negative PnL alongside deteriorating IC values reinforces the real impact of this degradation. Despite the regime still being "HIGH_VOL," the feature no longer responds well to volatility conditions it previously performed in, signaling a loss of efficacy. There is no obvious sign of data quality issues or regime mismatch since regime remained stable.


## Evidence

  - ic_5d dropped from ~0.11 on 2025-02-01 to 0.031 on 2025-02-03.
  - ic_ratio declined from above 2.0 to below 0.9, triggering the ic_decay alert.
  - PnL turned consistently negative and worsened as IC deteriorated.
  - Market regime remained "HIGH_VOL," so regime shift is unlikely to explain degradation alone.

## Proposed Fix

```
# Proposal to improve feature robustness:
# Add adaptive volatility scaling and include a regime filter to recalibrate VWAP deviation thresholds,
# e.g. expression adjustment:
"mean_reversion_vwap_dev_scaled = mean_reversion_vwap_deviation / rolling_std(price, window=20); 
 mean_reversion_signal = IF(regime == 'HIGH_VOL', clipped_scaled_signal(mean_reversion_vwap_dev_scaled), 0)"

```

## Learning Document

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

