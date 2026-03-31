# Midas

**Compound Engineering Framework for Alpha Feature Research**

> *Each iteration makes the next one cheaper.*

Midas is a dual-loop system for discovering and maintaining predictive alpha features on crypto perpetual futures. An **Offline Loop** searches for new signals using an LLM as a quant researcher. An **Online Loop** monitors live features, diagnoses degradation, and fires kill signals. Both loops write structured learning documents to a shared knowledge base — so every failure and every success permanently improves future research.

→ **[Full visual documentation](./README.html)** (open in browser for the pipeline diagrams)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE LOOP  (Discovery)                    │
│                                                                 │
│   Plan ──► Write ──► Assess ──► Learn ──► candidates/          │
│    ▲          │         │         │                             │
│    └──────────┴─ refine ┘         └──► knowledge base          │
│                  (if failing)                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ promote()
┌─────────────────────────▼───────────────────────────────────────┐
│                    ONLINE LOOP  (Improvement)                   │
│                                                                 │
│   Deploy ──► Monitor ──► Diagnose ──► Learn ──► Kill / Fix     │
│     ▲                        │          │                       │
│     └────────────────────────┘          └──► knowledge base    │
│                  (fix & redeploy)                               │
└─────────────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼────────────┐
              │     Knowledge Base     │
              │  offline/ · online/    │
              │  skills/ · regimes/    │
              │  thresholds.json       │
              └────────────────────────┘
```

---

## Key Design Decisions

- **No LLM in the inner loop** — all metric computation is vectorised numpy/pandas. Hundreds of expressions can be scored per second. The LLM is called only for Plan, Generate, Refine, Learn, and Diagnose.
- **DSL validation before compute** — `DSLValidator` catches unknown operators, negative lookbacks, unbalanced parentheses, and nesting depth > 5 before an expression ever reaches the feature engine.
- **Structured learning, not chat history** — every loop run writes a typed `LearningDocument` (pattern identified, why it worked/failed, suggestions). These load verbatim into future prompts with no summarisation loss.
- **Six parallel evaluation agents** — each agent scores one quality dimension and returns concrete improvement suggestions that feed directly into the Refine prompt.
- **Filesystem as source of truth** — plain markdown files in a predictable directory tree. Git-trackable, human-readable, no database required.

---

## Installation

```bash
pip install anthropic pandas numpy scikit-learn pyyaml
```

Clone this repo, then import directly:

```python
import sys
sys.path.insert(0, "/path/to/midas")
from midas.factory import create_midas
```

---

## Quickstart

```python
from midas.factory import create_midas

# Bootstrap — creates the KB directory tree and seeds default skills/prompts
midas = create_midas(
    kb_path = "./midas-kb",
    api_key = "sk-ant-...",      # or set ANTHROPIC_API_KEY env var
)

# ── Offline loop ──────────────────────────────────────────────────────────
def my_data_fn():
    # Return: (compute_fn, forward_returns_df, regime_series)
    # compute_fn takes a DSL expression string → pd.Series of feature values
    return engine.compute_feature, engine.get_fwd_returns(), engine.get_regimes()

learning = midas.offline.run(
    research_goal    = "Short-term mean-reversion on VWAP deviation",
    existing_factors = midas.promoter.list_deployed(),
    data_fn          = my_data_fn,
    regime           = "HIGH_VOL",
)
print(learning.result)             # "success" or "failed"
print(learning.pattern_identified) # saved to KB for future sessions

# ── Promote a winning candidate to production ─────────────────────────────
midas.promoter.promote("vwap_zscore_mean_rev")

# ── Online loop ───────────────────────────────────────────────────────────
monitor = midas.build_online(
    feature_names = midas.promoter.list_deployed(),
    on_kill       = lambda name: engine.disable_feature(name),
)

async for bar in engine.live_feed():
    await monitor.process_update(
        timestamp      = bar.ts,
        feature_values = bar.signals,        # {feature_name: float}
        forward_return = bar.ret_1h,
        regime         = bar.regime,
        market_context = {"btc_vol": bar.btc_vol},
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
python -m midas promote vwap_zscore_mean_rev

# Retire a live feature with a reason
python -m midas demote vwap_zscore_mean_rev --reason "IC < 0 in LOW_VOL regime"

# Reject a candidate back to archived
python -m midas reject noisy_ob_imbalance --reason "overfit ratio 2.4"

# Print the 5 most recent learning documents
python -m midas learnings --n 5
```

---

## Modules

| File | Lines | Responsibility |
|------|------:|----------------|
| `models.py` | 233 | All shared dataclasses — `EvaluationResult`, `MultiAgentResult`, `Alert`, `FeatureMetrics`, `LearningDocument`, `DailyReport`, `DiagnoseResult` |
| `kb.py` | 484 | Filesystem abstraction for the knowledge base. All reads/writes go through typed helpers. Seeds default DSL skill, prompt templates, and `thresholds.json` on first run |
| `evaluator.py` | 524 | `AlphaEvaluator` (vectorised metrics) + `MultiAgentEvaluator` (six agents in `ThreadPoolExecutor`) |
| `proposer.py` | 311 | `DSLValidator` (fast, no LLM) + `ExpressionProposer` (Plan / Generate / Refine via Anthropic API) |
| `loops.py` | 399 | `OfflineCompoundLoop` — Plan→Write→Assess→Learn, refine back-arc, writes `LearningDocument` to KB every run |
| `monitor.py` | 679 | `MonitorEngine` (rolling buffers) · `AlertEngine` (threshold rules) · `DiagnoseAgent` (async LLM) · `OnlineMonitor` (orchestrator) |
| `promoter.py` | 161 | `FeaturePromoter` — candidate→deployed→archived transitions with markdown state-patching |
| `factory.py` | 205 | `create_midas()` one-liner bootstrap · `Midas` container · CLI entry point |

---

## Evaluation Agents

`MultiAgentEvaluator` runs six agents in parallel. Each returns a verdict (pass/fail), a 0–1 score, and concrete improvement suggestions injected into the next Refine prompt.

| Agent | Checks | Pass criteria |
|-------|--------|---------------|
| `predictive_power` | Spearman IC, Information Ratio | `rank_ic ≥ 0.02`, `IR ≥ 0.5` |
| `decay_analysis` | Alpha half-life across horizons | `half_life ≥ 4h` |
| `trading_cost` | Turnover, effective IC after spread | `effective_ic > 0`, `turnover ≤ 0.8` |
| `diversification` | Correlation with deployed factors, marginal IC | `max_corr ≤ 0.70`, `marginal_ic > 0.01` |
| `overfit_detection` | In-sample vs out-of-sample IC ratio | `IS/OOS ≤ 1.5`, `OOS_IC > 0.01` |
| `regime_robustness` | IC variance across market regimes | `min_regime_IC > 0`, `IC_range < 0.05` |

Composite score weights: rank\_ic 30% · IR 20% · effective\_ic 20% · marginal\_ic 15% · OOS\_IC 15%.

---

## Online Alert Thresholds

| Alert | Trigger | Severity |
|-------|---------|----------|
| IC decay | `ic_5d / ic_30d < 0.70` | warning |
| IC decay | `ic_5d / ic_30d < 0.50` | **critical** → LLM diagnosis |
| Slippage | `realized / expected > 1.5×` | warning |
| Slippage | `realized / expected > 2.0×` | **critical** → LLM diagnosis |
| PnL drawdown | `1-day contribution < −50 bps` | warning |

Critical alerts trigger an async `DiagnoseAgent` call that writes a `learning_doc`, a `fix_proposal` (revised DSL expression), and a `kill_signal` to the knowledge base.

All thresholds are overridable at runtime via `alert_thresholds=` in `build_online()`.

---

## Knowledge Base Layout

```
midas-kb/
├── skills/
│   ├── midas-dsl.md          # DSL operator reference — seeded automatically
│   └── factor-patterns.md    # common alpha pattern catalogue
├── knowledge/
│   ├── features/
│   │   ├── deployed/         # live in production — monitored every bar
│   │   ├── candidates/       # passed backtest, awaiting promote()
│   │   └── archived/         # retired, failure analysis attached
│   ├── learnings/
│   │   ├── offline/          # YYYY-MM-DD-<pattern>.md
│   │   └── online/           # YYYY-MM-DD-<feature>-<alert_type>.md
│   ├── regimes/              # market regime documentation
│   └── thresholds.json       # pass/fail thresholds for all agents
├── proposer/prompts/
│   ├── plan.md               # editable — loaded fresh each session
│   ├── generate.md
│   └── refine.md
└── reports/
    ├── daily/                # YYYY-MM-DD.md
    └── diagnoses/            # per-alert LLM diagnosis reports
```

All prompt templates are seeded on first run and are fully editable — changes take effect immediately on the next loop run.

---

## Integrating Your Feature Engine

Midas has two integration seams:

**Offline** — pass a `data_fn` callable to `loop.run()`:

```python
def data_fn():
    def compute(expression: str) -> pd.Series:
        # Parse and evaluate the DSL expression against your data
        return your_engine.compute(expression)

    forward_returns = pd.DataFrame({
        "ret_1h":  ...,   # required
        "ret_4h":  ...,   # optional, for decay analysis
        "ret_8h":  ...,
        "ret_24h": ...,
        "ret_48h": ...,
    }, index=timestamps)

    regime_labels = pd.Series(["HIGH_VOL", "LOW_VOL", ...], index=timestamps)

    return compute, forward_returns, regime_labels
```

**Online** — call `process_update()` on each completed bar:

```python
await monitor.process_update(
    timestamp      = datetime.utcnow(),
    feature_values = {"signal_a": 0.42, "signal_b": -0.17},  # {name: float}
    forward_return = 0.0031,    # 1-hour forward return as a fraction
    fills          = {"slippage_bps": 3.2},                   # optional
    regime         = "HIGH_VOL",
    market_context = {},        # any dict, passed verbatim to DiagnoseAgent
)
```

---

## Running the Tests

```bash
PYTHONPATH=. python midas/tests/test_integration.py
```

All seven tests run against a mock LLM — no API key required:

```
  ✓  DSL Validator
  ✓  Static Evaluator
  ✓  Multi-Agent Evaluator
  ✓  Knowledge Base
  ✓  Feature Promoter
  ✓  Offline Compound Loop
  ✓  Online Monitor

  Results: 7 passed, 0 failed
```

---

## Configuration

`OfflineLoopConfig` controls the search budget:

```python
from midas.loops import OfflineLoopConfig

config = OfflineLoopConfig(
    max_iterations      = 10,   # max Plan→Assess cycles per session
    candidates_per_iter = 3,    # expressions generated per LLM call
    model               = "claude-sonnet-4-20250514",
    max_tokens          = 2000,
    verbose             = True, # print iteration log to stdout
)

midas = create_midas(offline_config=config, ...)
```

Pass custom thresholds directly to the evaluator:

```python
midas = create_midas(
    existing_features = deployed_factor_df,  # for correlation checks
    spread_cost_bps   = 3.0,                 # your actual spread assumption
)
```

---

## Requirements

- Python 3.11+
- `anthropic` — LLM calls (Plan, Generate, Refine, Diagnose, Learn)
- `pandas`, `numpy` — vectorised metric computation
- `scikit-learn` — marginal IC residualisation
- `pyyaml` — LLM response parsing

---

*Midas v0.1.0 — 3,030 lines — 7/7 tests passing*