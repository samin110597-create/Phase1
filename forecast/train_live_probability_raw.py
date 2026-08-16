from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import forecast.train_live_probability as trainer


def raw_daily_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Use unadjusted OHLC so daily labels/features stay on the same price scale as Twelve Data intraday bars."""
    raw = yf.download(
        symbols,
        start='2019-01-01',
        auto_adjust=False,
        actions=False,
        group_by='ticker',
        threads=True,
        progress=False,
    )
    out = {}
    for symbol in symbols:
        try:
            d = raw[symbol].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            d = d.rename(columns=str.lower).dropna(subset=['close'])
            d.index = pd.to_datetime(d.index).tz_localize(None).astype('datetime64[ns]')
            out[symbol] = d[['open', 'high', 'low', 'close', 'volume']].copy()
        except Exception:
            out[symbol] = pd.DataFrame()
    return out


trainer.daily_history = raw_daily_history

if __name__ == '__main__':
    trainer.main()
