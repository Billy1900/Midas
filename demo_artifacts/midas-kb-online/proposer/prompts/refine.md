# Alpha Expression Refinement Prompt

You are iterating on an alpha feature based on multi-agent evaluation feedback.

## Original Expression
{{expression}}

## Evaluation Results (JSON)
{{evaluation_result}}

## Blocking Issues
{{blocking_issues}}

## Agent Suggestions
{{suggestions}}

## Common Fixes
- High turnover          → wrap in ema() to smooth
- Short half-life        → increase lookback or add lag
- High existing corr     → residualise with cs_neutralize or sub out the correlated component
- Regime-dependent       → add if_else regime filter
- Overfit (ratio > 1.5)  → simplify — remove parameters or reduce nesting

## Output (YAML only)
```yaml
refined_expression: |
  the_improved_dsl_expression

changes_made:
  - "<change> — <reason>"

expected_improvement:
  - "<metric> should improve because <reason>"
```
