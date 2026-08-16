# Phase1 Live Forecast Engine V2 backend

This folder is intentionally separate from GitHub Pages so market-data API keys never appear in browser code.

## Runtime
- `GET /api/quote?symbols=AAPL,NVDA,MSFT` — secure Finnhub live quote pull for up to 3 symbols.
- `GET /api/forecast?symbols=AAPL,NVDA,MSFT` — provisional live research state using Finnhub quote + Twelve Data 15m + rolling 4h + daily + weekly + SPY context.
- Quote endpoint is designed for ~10-second UI refreshes.
- Forecast endpoint is designed for ~60-second refreshes.

## Required server environment variables
- `FINNHUB_API_KEY`
- `TWELVE_DATA_API_KEY`

Do not put secret values in this repository or in `docs/`.

## Vercel project setup
Import `samin110597-create/Phase1` and set the project Root Directory to `live-backend`. Add the two environment variables in Vercel Project Settings, deploy, then place only the public deployment base URL in `docs/data/live_backend.json` as `api_base`.

## Important
The current directional values are **provisional live scores**, not calibrated probabilities. The accuracy/calibration pipeline is intentionally separate.
