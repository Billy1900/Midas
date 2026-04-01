# Midas

**Compound Engineering Framework for Alpha Feature Research**

> *Each iteration makes the next one cheaper.*

Midas is a dual-loop system for discovering and maintaining predictive alpha features on crypto perpetual futures.

- The **Offline Loop** searches for new signals using an LLM as a quant researcher.
- The **Online Loop** monitors live features, diagnoses degradation, and fires kill signals.
- Both loops write structured learning documents to a shared knowledge base, so every failure and every success can improve future research.

[Full visual documentation](https://billy1900.github.io/Midas/)

---

## Architecture

```text
+--------------------------------------------------------------+
|                  OFFLINE LOOP (Discovery)                    |
|                                                              |
|  Plan -> Write -> Assess -> Learn -> candidates/             |
|              ^                    |                           |
|              |                    v                           |
|           Refine <--------- knowledge base                   |
+------------------------------+-------------------------------+
                               |
                               | promote()
                               v
+--------------------------------------------------------------+
|                  ONLINE LOOP (Improvement)                   |
|                                                              |
|  Deploy -> Monitor -> Diagnose -> Learn -> Kill / Fix        |
|                        |                    ^                 |
|                        v                    |                 |
|                   knowledge base ---- fix & redeploy         |
+------------------------------+-------------------------------+
                               |
                               v
                    +-------------------------+
                    |      Knowledge Base     |
                    | offline/  online/       |
                    | skills/   regimes/      |
                    | thresholds.json         |
                    +-------------------------+
```

---

## Install

```bash
pip install -e .
```

This repository now installs as a normal Python package and supports:

- `python -m midas ...`
- `from midas import create_midas`

---

## Demo Outputs

This repository intentionally keeps two example output directories:

- `demo_artifacts/`
- `midas-kb/`

They are included as **demo and documentation artifacts**, not as production data directories.
They show what the framework writes to disk during offline and online runs.

---

## Quick Demo

Run the bundled end-to-end demo without any API key:

```bash
python -m midas demo
```

This demo:

1. runs one offline discovery session on synthetic data
2. runs one online monitoring session on synthetic bars
3. writes a Markdown report to `demo_artifacts/DEMO_REPORT.md`

You can choose a custom report path:

```bash
python -m midas demo --report-path ./artifacts/demo_report.md
```

---

## Using OpenAI

Set your key:

```bash
set OPENAI_API_KEY=your_key_here
```

Then run:

```bash
python -m midas demo --provider openai
```

Optional model override:

```bash
python -m midas demo --provider openai --model gpt-4.1-mini
```

You can also pass the key directly:

```bash
python -m midas demo --provider openai --api-key your_key_here
```

---

## Using Anthropic

```bash
set ANTHROPIC_API_KEY=your_key_here
python -m midas demo --provider anthropic
```

---

## Quickstart

```python
from midas import create_midas

# Bootstrap - creates the KB directory tree and seeds default skills/prompts
midas = create_midas(
    kb_path="./midas-kb",
    provider="openai",   # or "anthropic"
    api_key="...",       # or use env vars
)

# Offline loop
def my_data_fn():
    # Return: (compute_fn, forward_returns_df, regime_series)
    # compute_fn takes a DSL expression string -> pd.Series of feature values
    return engine.compute_feature, engine.get_fwd_returns(), engine.get_regimes()

learning = midas.offline.run(
    research_goal="Short-term mean-reversion on VWAP deviation",
    existing_factors=midas.promoter.list_deployed(),
    data_fn=my_data_fn,
    regime="HIGH_VOL",
)
print(learning.result)             # "success" or "failed"
print(learning.pattern_identified) # saved to KB for future sessions

# Promote a winning candidate to production
midas.promoter.promote("vwap_zscore_mean_rev")

# Online loop
monitor = midas.build_online(
    feature_names=midas.promoter.list_deployed(),
    on_kill=lambda name: engine.disable_feature(name),
)

async for bar in engine.live_feed():
    await monitor.process_update(
        timestamp=bar.ts,
        feature_values=bar.signals,        # {feature_name: float}
        forward_return=bar.ret_1h,
        regime=bar.regime,
        market_context={"btc_vol": bar.btc_vol},
    )

# End of day
report = monitor.generate_daily_report("2025-01-15")
```

---

## CLI

```bash
# View the current feature pipeline (candidates / deployed / archived)
python -m midas status --kb ./midas-kb

# Promote a candidate to deployed
python -m midas promote vwap_zscore_mean_rev --kb ./midas-kb

# Retire a live feature with a reason
python -m midas demote vwap_zscore_mean_rev --kb ./midas-kb --reason "IC < 0 in LOW_VOL regime"

# Reject a candidate back to archived
python -m midas reject noisy_ob_imbalance --kb ./midas-kb --reason "overfit ratio 2.4"

# Print the 5 most recent learning documents
python -m midas learnings --kb ./midas-kb --n 5

# Run the bundled demo
python -m midas demo
```

---

## Package Layout

| File | Responsibility |
|------|----------------|
| `midas/models.py` | Shared dataclasses: `EvaluationResult`, `MultiAgentResult`, `Alert`, `FeatureMetrics`, `LearningDocument`, `DailyReport`, `DiagnoseResult` |
| `midas/kb.py` | Filesystem abstraction for the knowledge base. All reads/writes go through typed helpers. Seeds DSL skill, prompt templates, and `thresholds.json` on first run |
| `midas/evaluator.py` | `AlphaEvaluator` (vectorised metrics) + `MultiAgentEvaluator` (six agents in `ThreadPoolExecutor`) |
| `midas/proposer.py` | `DSLValidator` (fast, no LLM) + `ExpressionProposer` (provider-backed Plan / Generate / Refine flow) |
| `midas/loops.py` | `OfflineCompoundLoop` - Plan -> Write -> Assess -> Learn, with refinement and KB persistence |
| `midas/monitor.py` | `MonitorEngine` (rolling buffers), `AlertEngine` (threshold rules), `DiagnoseAgent` (async LLM), `OnlineMonitor` (orchestrator) |
| `midas/promoter.py` | `FeaturePromoter` - candidate -> deployed -> archived transitions with markdown state updates |
| `midas/factory.py` | `create_midas()` bootstrap, `Midas` container, CLI wiring |
| `midas/llm.py` | OpenAI / Anthropic client resolution and compatibility adapter |
| `midas/demo.py` | Bundled synthetic demo and report generation |

---

## Evaluation Agents

`MultiAgentEvaluator` runs six agents in parallel. Each returns a verdict, a `0..1` score, and concrete suggestions that can be injected into the next Refine prompt.

| Agent | Checks | Pass criteria |
|-------|--------|---------------|
| `predictive_power` | Spearman IC, Information Ratio | `rank_ic >= 0.02`, `IR >= 0.5` |
| `decay_analysis` | Alpha half-life across horizons | `half_life >= 4h` |
| `trading_cost` | Turnover, effective IC after spread | `effective_ic > 0`, `turnover <= 0.8` |
| `diversification` | Correlation with deployed factors, marginal IC | `max_corr <= 0.70`, `marginal_ic > 0.01` |
| `overfit_detection` | In-sample vs out-of-sample IC ratio | `IS/OOS <= 1.5`, `OOS_IC > 0.01` |
| `regime_robustness` | IC variance across market regimes | `min_regime_IC > 0`, `IC_range < 0.05` |

Composite score weights:

- `rank_ic` 30%
- `IR` 20%
- `effective_ic` 20%
- `marginal_ic` 15%
- `OOS_IC` 15%

---

## Online Alert Thresholds

| Alert | Trigger | Severity |
|-------|---------|----------|
| IC decay | `ic_5d / ic_30d < 0.70` | warning |
| IC decay | `ic_5d / ic_30d < 0.50` | **critical** -> LLM diagnosis |
| Slippage | `realized / expected > 1.5x` | warning |
| Slippage | `realized / expected > 2.0x` | **critical** -> LLM diagnosis |
| PnL drawdown | `1-day contribution < -50 bps` | warning |

Critical alerts trigger an async `DiagnoseAgent` call that writes:

- a `learning_doc`
- a `fix_proposal`
- a `kill_signal`

All thresholds are overridable at runtime via `alert_thresholds=` in `build_online()`.

---

## Knowledge Base Layout

```text
midas-kb/
|-- skills/
|   |-- midas-dsl.md
|   `-- factor-patterns.md
|-- knowledge/
|   |-- features/
|   |   |-- deployed/
|   |   |-- candidates/
|   |   `-- archived/
|   |-- learnings/
|   |   |-- offline/
|   |   `-- online/
|   |-- regimes/
|   `-- thresholds.json
|-- proposer/prompts/
|   |-- plan.md
|   |-- generate.md
|   `-- refine.md
`-- reports/
    |-- daily/
    `-- diagnoses/
```

All prompt templates are seeded on first run and are fully editable. Changes take effect on the next loop run.

---

## Integrating Your Feature Engine

Midas depends on the interface:

```python
compute(expression: str) -> pd.Series
```

You can wrap any engine behind that interface, whether it is:

- a pandas-based evaluator
- a custom feature engine
- a Spark job
- another internal signal service

### Offline

Pass a `data_fn` callable to `loop.run()`:

```python
def data_fn():
    def compute(expression: str) -> pd.Series:
        return your_engine.compute(expression)

    forward_returns = pd.DataFrame({
        "ret_1h":  ...,
        "ret_4h":  ...,
        "ret_8h":  ...,
        "ret_24h": ...,
        "ret_48h": ...,
    }, index=timestamps)

    regime_labels = pd.Series(["HIGH_VOL", "LOW_VOL", ...], index=timestamps)

    return compute, forward_returns, regime_labels
```

### Online

Call `process_update()` on each completed bar:

```python
await monitor.process_update(
    timestamp=datetime.utcnow(),
    feature_values={"signal_a": 0.42, "signal_b": -0.17},
    forward_return=0.0031,
    fills={"slippage_bps": 3.2},   # optional
    regime="HIGH_VOL",
    market_context={},
)
```

---

## Running the Tests

```bash
python test_integration.py
```

All seven tests run against a mock LLM, so no API key is required.

---

## Configuration

`OfflineLoopConfig` controls the offline search budget:

```python
from midas.loops import OfflineLoopConfig

config = OfflineLoopConfig(
    max_iterations=10,
    candidates_per_iter=3,
    model="gpt-4.1-mini",   # or a Claude model
    max_tokens=2000,
    verbose=True,
)

midas = create_midas(
    provider="openai",
    offline_config=config,
)
```

Pass custom evaluation assumptions directly when bootstrapping:

```python
midas = create_midas(
    existing_features=deployed_factor_df,
    spread_cost_bps=3.0,
)
```

---

## Current Scope

Midas is a research framework, not a full exchange execution engine.

It now supports:

- installable package layout
- CLI usage
- OpenAI-backed runs
- Anthropic-backed runs
- mock demo runs with no API key

You still need to provide your own real factor engine and real market data if you want to use it beyond the bundled synthetic demo.
