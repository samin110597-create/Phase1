# Phase1 — Institutional Stock Intelligence

Public stock-analysis dashboard with two views:

1. **Institutional Screener** — ranks liquid US stocks by trend quality, relative strength, technical structure, risk and observable accumulation/distribution proxies.
2. **Momentum Radar** — ranks stocks by momentum strength while showing a separate **confidence score** and evidence behind the score.

## Security architecture

API keys are never placed in `docs/`, browser JavaScript, URLs, localStorage or committed JSON. Market-data collection runs in GitHub Actions and reads credentials only from repository/environment secrets. The public website reads sanitized `docs/data/latest.json`.

Supported secret names (same provider convention as the earlier projects):

- `TWELVE_DATA_API_KEY`
- `FINNHUB_API_KEY`
- `FMP_API_KEY`
- `POLYGON_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

The scanner still runs with public/fallback market data if those secrets are unavailable; cross-source validation is then marked unavailable and confidence is reduced rather than faked.

## What “institutional” means here

The app does **not** claim to see private institutional order flow. Its Institutional Proxy score uses observable evidence: benchmark-relative strength, OBV/accumulation behaviour, up-volume vs down-volume, price/volume breakout quality, moving-average structure, volatility contraction/expansion and liquidity. Any reported ownership/filing data can be added only when a configured provider supplies it.

## Accuracy rules

- Momentum and confidence are different metrics.
- Confidence is penalized for missing/stale data or cross-source disagreement.
- No high-conviction label when data quality is low.
- Signals are analytical rankings, not guaranteed predictions.
- Future ML probability/target outputs must pass out-of-sample calibration/stability gates before the UI can call them verified.

## Structure

- `scripts/scan.py` — data collection + indicators + institutional/momentum/confidence scoring
- `docs/index.html` — public dashboard
- `docs/data/latest.json` — sanitized generated output
- `.github/workflows/scan.yml` — scheduled scanner
- `.github/workflows/pages.yml` — GitHub Pages deployment

## Schedule

The scan is scheduled for weekdays after the US market close and can also be run manually from GitHub Actions.
