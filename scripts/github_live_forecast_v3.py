from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v2 as engine

_original_quote = base.finnhub_quote


def resilient_quote(symbol: str):
    last_error = None
    for delay in (0.0, 0.8, 2.0):
        if delay:
            time.sleep(delay)
        try:
            return _original_quote(symbol)
        except Exception as exc:
            last_error = exc
    raise last_error if last_error else RuntimeError('quote failed')


# Patch only quote acquisition. The model, features, calibration status and
# backtest rules are unchanged by this resilience layer.
base.finnhub_quote = resilient_quote

if __name__ == '__main__':
    engine.main()
