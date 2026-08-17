from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v4 as v4
import scripts.github_live_forecast_v5 as v5

OUT = Path('docs/data/live_forecast.json')
SOURCE_USED: dict[str, str] = {}

_base_twelve = base.twelve_15m
_v4_recent = v4.recent_15m


def yahoo_15m(symbol: str, outputsize: int = 220) -> pd.DataFrame:
    """Resilient secondary intraday source used only when Twelve Data is unavailable/rate-limited."""
    raw = yf.download(
        symbol,
        period='60d',
        interval='15m',
        auto_adjust=False,
        actions=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError('Yahoo Finance 15m fallback returned no bars')

    x = raw.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # yfinance commonly returns (field, ticker) columns for a single ticker.
        if symbol in x.columns.get_level_values(-1):
            x = x.xs(symbol, level=-1, axis=1)
        else:
            x.columns = x.columns.get_level_values(0)

    x = x.rename(columns=lambda c: str(c).lower().replace(' ', '_'))
    if 'close' not in x.columns:
        raise RuntimeError('Yahoo Finance 15m fallback missing close')

    x = x.reset_index()
    dt_col = next((c for c in x.columns if str(c).lower() in ('datetime', 'date')), x.columns[0])
    x = x.rename(columns={dt_col: 'datetime'})
    x['datetime'] = pd.to_datetime(x['datetime'], errors='coerce')
    try:
        if x['datetime'].dt.tz is not None:
            x['datetime'] = x['datetime'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    except Exception:
        pass

    for c in ('open', 'high', 'low', 'close', 'volume'):
        if c not in x.columns:
            x[c] = 0.0 if c == 'volume' else pd.NA
        x[c] = pd.to_numeric(x[c], errors='coerce')

    x = x.dropna(subset=['datetime', 'open', 'high', 'low', 'close']).sort_values('datetime')
    tm = x['datetime'].dt.strftime('%H:%M')
    x = x[(tm >= '09:30') & (tm <= '16:00')]
    x = x[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(max(220, int(outputsize))).reset_index(drop=True)
    if len(x) < 80:
        raise RuntimeError(f'Yahoo Finance 15m fallback has insufficient bars: {len(x)}')
    SOURCE_USED[symbol] = 'Yahoo Finance 15m fallback'
    return x


def resilient_base_15m(symbol: str, outputsize: int = 220) -> pd.DataFrame:
    try:
        x = _base_twelve(symbol, outputsize)
        SOURCE_USED[symbol] = 'Twelve Data 15m'
        return x
    except Exception:
        return yahoo_15m(symbol, outputsize)


def resilient_v4_15m(symbol: str, outputsize: int = 220) -> pd.DataFrame:
    try:
        x = _v4_recent(symbol, outputsize)
        SOURCE_USED[symbol] = 'Twelve Data 15m'
        return x
    except Exception:
        return yahoo_15m(symbol, outputsize)


# Patch both the current legacy path and the V3/V4 auto-upgrade path.
base.twelve_15m = resilient_base_15m
v4.recent_15m = resilient_v4_15m


def main():
    v5.main()
    payload = json.loads(OUT.read_text())

    fallback_symbols = []
    for row in payload.get('rows', []):
        symbol = row.get('symbol')
        src = SOURCE_USED.get(symbol)
        if src:
            row['intraday_source'] = src
            if 'fallback' in src.lower():
                fallback_symbols.append(symbol)

    payload.setdefault('sources', {})['intraday'] = 'Twelve Data 15m primary; Yahoo Finance 15m automatic fallback on rate-limit/provider failure'
    payload['data_resilience'] = {
        'blank_dashboard_guard': True,
        'primary_intraday_source': 'Twelve Data 15m',
        'secondary_intraday_source': 'Yahoo Finance 15m fallback',
        'fallback_symbols': sorted(set(fallback_symbols)),
        'policy': 'A temporary Twelve Data 429/provider failure must not erase otherwise usable ticker rows. Fallback usage is explicitly labeled and never presented as the primary source.'
    }
    if fallback_symbols:
        payload['intraday_provider_state'] = 'DEGRADED_FALLBACK_ACTIVE'
    else:
        payload['intraday_provider_state'] = 'PRIMARY_SOURCE_OK'

    OUT.write_text(json.dumps(payload, separators=(',', ':')))


if __name__ == '__main__':
    main()
