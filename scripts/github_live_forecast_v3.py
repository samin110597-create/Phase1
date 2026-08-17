from __future__ import annotations

import time

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


# Patch only the quote acquisition used by the existing engine. This keeps all
# scoring/backtest logic unchanged while making transient provider failures less
# likely to remove names from the live universe.
base.finnhub_quote = resilient_quote

if __name__ == '__main__':
    engine.main()
