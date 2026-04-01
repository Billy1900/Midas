# Common Alpha Factor Patterns

## Momentum
- Short-term price momentum (1–8h)
- VWAP deviation momentum
- Volume-weighted price momentum

## Mean Reversion
- RSI extremes
- Bollinger Band %B
- Z-score of returns vs rolling baseline

## Microstructure
- Bid/ask imbalance
- Trade-flow imbalance (taker buy vs sell)
- Volume surprise (actual vs expected)

## Carry / Funding
- Perpetual funding rate as carry signal
- Open-interest change as positioning signal

## Regime Filters
- Use `if_else` to gate signal on ATR regime
- Separate bull/bear factor weights via regime label
