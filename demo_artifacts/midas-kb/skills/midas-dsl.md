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
