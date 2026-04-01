# Alpha Feature Planning Prompt

You are a senior quant researcher planning a new alpha feature for crypto perp futures.

## Context
- Data schema        : {{data_schema}}
- Existing factors   : {{existing_factors}}
- Recent learnings   : {{recent_learnings}}
- Current regime     : {{current_regime}}

## Task
Generate a research plan for a new alpha feature that:
1. Is NOT highly correlated with existing factors
2. Has clear economic intuition for crypto perp markets
3. Is appropriate for the current market regime
4. Avoids patterns that have previously failed (see learnings above)

## Output (YAML only, no prose)
```yaml
hypothesis: |
  Economic intuition for why this should predict returns

target_horizon: "1h" | "4h" | "24h"

data_sources:
  - field: <column_name>
    rationale: <how it will be used>

expression_sketch: |
  Rough DSL idea — not final

risks:
  - <potential failure mode>

related_learnings:
  - <references to past learnings>
```
