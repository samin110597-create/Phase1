# Phase1 — Q-State Research & Validation Lab

**Phase1 is no longer a production forecasting model.** The production system is Q-State in `stock-truth-v2`.

Phase1 exists to improve that one model through controlled experiments, walk-forward validation, calibration checks, regime tests, failure analysis, and promotion reports. It must not publish a competing live BUY/SELL forecast, price target, or probability.

## Single-model relationship

```
Q-State Deno canonical data
        ↓
Phase1 candidate research
        ↓
chronological walk-forward + calibration + regime validation
        ↓
promotion report
        ↓
validated change merged into Q-State
```

The public-facing production forecast remains Q-State only.

## Research surfaces

1. **Institutional Screener** — descriptive ranking of liquid US stocks by trend quality, relative strength, technical structure, risk and observable accumulation/distribution proxies.
2. **Momentum Radar** — descriptive momentum ranking with evidence and confidence diagnostics.
3. **Candidate forecast experiments** — manual/research-only experiments. Their outputs are not production forecasts unless the validated logic is later merged into Q-State.

## Canonical data source

New research should use the same Deno market-data contract as Q-State whenever possible. `scripts/qstate_data_client.py` provides a small client for:

- `/health`
- `/v1/quote?symbol=...`
- `/v1/market?symbol=...&timeframe=...`

Set `QSTATE_API_BASE` to the deployed Q-State Deno endpoint when overriding the default.

## Reproducibility

Persisted scikit-learn artifacts in this repository were created with scikit-learn 1.9.0. The runtime is pinned to that version to prevent silent cross-version deserialization drift.

## Security architecture

API keys are never placed in `docs/`, browser JavaScript, URLs, localStorage or committed JSON. Research collection reads credentials only from repository/environment secrets. The public website reads sanitized outputs.

Supported secret names:

- `TWELVE_DATA_API_KEY`
- `FINNHUB_API_KEY`
- `FMP_API_KEY`
- `POLYGON_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

## Accuracy rules

- Research output must be labeled research-only until promoted into Q-State.
- Momentum/confidence scores are not predictive probabilities.
- Missing or stale data reduces confidence; it is never silently filled with fabricated facts.
- A candidate model must beat the current Q-State baseline out of sample before promotion.
- Chronological walk-forward validation, calibration, regime stability, and prospective evidence are required for predictive probabilities.
- No candidate experiment may rewrite historical signals after outcomes are known.

## Workflow policy

`.github/workflows/intraday.yml` is manual-only. It can generate research evidence and update the research dashboard, but it does not run as a competing scheduled production forecaster.

The scheduled production forecasting responsibility belongs to Q-State in `stock-truth-v2`.
