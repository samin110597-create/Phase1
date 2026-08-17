from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import forecast.build_intraday_live_like_dataset as base
import forecast.build_intraday_live_like_dataset_v3 as fast


def cache_only_fetch(symbol: str) -> pd.DataFrame:
    path = base.CACHE / f'{symbol.replace("^", "IDX_")}.csv.gz'
    if not path.exists():
        raise RuntimeError(f'cache missing for {symbol}')
    x = pd.read_csv(path, compression='gzip', parse_dates=['datetime'])
    x = base.parse_bars(x.to_dict('records'))
    x = x[x['datetime'] >= base.START]
    tm = x['datetime'].dt.strftime('%H:%M')
    x = x[(tm >= '09:30') & (tm <= '15:45')].reset_index(drop=True)
    if x.empty:
        raise RuntimeError(f'cache empty for {symbol}')
    return x


base.fetch_15m = cache_only_fetch
base.make_snapshots = fast.fast_make_snapshots

if __name__ == '__main__':
    base.main()
