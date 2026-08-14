from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

SUMMARY = Path('docs/data/training_summary.json')
OUT = Path('forecast/data/training_v1.csv.gz')
LATEST = Path('docs/data/latest.json')
START_DATE = os.getenv('PHASE1_TRAIN_START', '2020-01-01')
MAX_SYMBOLS = int(os.getenv('PHASE1_TRAIN_SYMBOLS', '12'))
TWELVE_KEY = os.getenv('TWELVE_DATA_API_KEY')
TIMEOUT = 20
REQUESTS_PER_MINUTE = 7


def get_json(url: str):
    req = Request(url, headers={'User-Agent': 'Phase1-Forecast-Research/1.0'})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode('utf-8'))


class RateLimiter:
    def __init__(self, limit=REQUESTS_PER_MINUTE):
        self.limit = limit
        self.count = 0
        self.window = time.monotonic()

    def wait(self):
        elapsed = time.monotonic() - self.window
        if elapsed >= 61:
            self.count = 0
            self.window = time.monotonic()
        if self.count >= self.limit:
            sleep_for = max(0, 61 - elapsed)
            print(f'rate-limit guard: sleeping {sleep_for:.1f}s')
            time.sleep(sleep_for)
            self.count = 0
            self.window = time.monotonic()
        self.count += 1


LIMITER = RateLimiter()


def fetch_15m(symbol: str) -> pd.DataFrame:
    if not TWELVE_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY is not configured')
    target = pd.Timestamp(START_DATE)
    end_date = None
    pieces = []
    seen_oldest = None
    for page in range(20):
        LIMITER.wait()
        params = {'symbol': symbol, 'interval': '15min', 'outputsize': 5000, 'apikey': TWELVE_KEY}
        if end_date:
            params['end_date'] = end_date
        d = get_json('https://api.twelvedata.com/time_series?' + urlencode(params))
        vals = d.get('values') if isinstance(d, dict) else None
        if not isinstance(vals, list) or not vals:
            print(symbol, 'stopped:', (d.get('message') if isinstance(d, dict) else 'no response'))
            break
        x = pd.DataFrame(vals)
        if 'datetime' not in x:
            break
        x['datetime'] = pd.to_datetime(x['datetime'], errors='coerce')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            x[c] = pd.to_numeric(x.get(c), errors='coerce')
        x = x.dropna(subset=['datetime', 'open', 'high', 'low', 'close']).sort_values('datetime')
        if x.empty:
            break
        pieces.append(x[['datetime', 'open', 'high', 'low', 'close', 'volume']])
        oldest = x['datetime'].min()
        print(symbol, 'page', page + 1, 'bars', len(x), 'oldest', oldest)
        if oldest <= target:
            break
        if seen_oldest is not None and oldest >= seen_oldest:
            break
        seen_oldest = oldest
        end_date = (oldest - pd.Timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%S')
    if not pieces:
        return pd.DataFrame()
    z = pd.concat(pieces, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
    z = z[z['datetime'] >= target]
    # Regular US session only; historical feature timestamps are kept in exchange-local time.
    t = z['datetime'].dt.time
    z = z[(t >= datetime.strptime('09:30', '%H:%M').time()) & (t <= datetime.strptime('15:45', '%H:%M').time())]
    return z.reset_index(drop=True)


def rsi(s: pd.Series, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def feature_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d = df.copy()
    c = d['close']; v = d['volume'].fillna(0)
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean(); macd = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean(); ema50 = c.ewm(span=50, adjust=False).mean()
    prev = c.shift(1); tr = pd.concat([(d['high']-d['low']), (d['high']-prev).abs(), (d['low']-prev).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    typical = (d['high'] + d['low'] + d['close']) / 3
    vwap20 = (typical * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    vol20 = v.rolling(20).mean()
    out = pd.DataFrame(index=d.index)
    out[f'{prefix}_ret3'] = c.pct_change(3)
    out[f'{prefix}_ret10'] = c.pct_change(10)
    out[f'{prefix}_rsi14'] = rsi(c)
    out[f'{prefix}_above_ema20'] = (c > ema20).astype(float)
    out[f'{prefix}_ema20_gt_ema50'] = (ema20 > ema50).astype(float)
    out[f'{prefix}_macd_delta_pct'] = (macd - sig) / c
    out[f'{prefix}_atr_pct'] = atr / c
    out[f'{prefix}_vol_ratio20'] = v / vol20.replace(0, np.nan)
    out[f'{prefix}_vwap20_dist'] = c / vwap20 - 1
    return out


def daily_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(symbols, start='2018-01-01', auto_adjust=True, actions=False, group_by='ticker', threads=True, progress=False)
    out = {}
    for s in symbols:
        try:
            x = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            x = x.rename(columns=str.lower).dropna(subset=['close'])
            out[s] = x
        except Exception:
            out[s] = pd.DataFrame()
    return out


def daily_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy(); x.index = pd.to_datetime(x.index).tz_localize(None)
    f = feature_frame(x.reset_index(drop=False).rename(columns={x.index.name or 'index':'datetime'}), 'day')
    f.index = x.index
    f['day_ret5'] = x['close'].pct_change(5)
    f['day_ret20'] = x['close'].pct_change(20)
    return f


def weekly_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy(); x.index = pd.to_datetime(x.index).tz_localize(None)
    w = pd.DataFrame({
        'open': x['open'].resample('W-FRI').first(),
        'high': x['high'].resample('W-FRI').max(),
        'low': x['low'].resample('W-FRI').min(),
        'close': x['close'].resample('W-FRI').last(),
        'volume': x['volume'].resample('W-FRI').sum(),
    }).dropna(subset=['close'])
    f = feature_frame(w.reset_index(drop=False).rename(columns={w.index.name or 'index':'datetime'}), 'week')
    f.index = w.index
    f['week_ret4'] = w['close'].pct_change(4)
    f['week_ret12'] = w['close'].pct_change(12)
    return f


def select_symbols():
    latest = json.loads(LATEST.read_text())
    rows = [x for x in latest.get('stocks', []) if float(x.get('price') or 0) >= 5 and float(x.get('avg_dollar_volume') or 0) >= 50_000_000]
    rows.sort(key=lambda x: float(x.get('avg_dollar_volume') or 0), reverse=True)
    return [x['symbol'] for x in rows[:MAX_SYMBOLS]]


def build_symbol(symbol: str, intraday: pd.DataFrame, daily: pd.DataFrame, spy_daily: pd.DataFrame, spy_15: pd.DataFrame) -> pd.DataFrame:
    if intraday.empty or daily.empty:
        return pd.DataFrame()
    z = intraday.copy(); z['date'] = z['datetime'].dt.normalize()
    f15 = feature_frame(z, 'm15')
    z = pd.concat([z, f15], axis=1)
    # Daily snapshot uses the final completed 15m bar of each session.
    last_idx = z.groupby('date')['datetime'].idxmax()
    snap = z.loc[last_idx].set_index('date').sort_index()
    # Rolling 4-hour structure from the last 16 regular-session 15m bars.
    c = z['close']; v = z['volume'].fillna(0)
    z['h4_ret'] = c.pct_change(16)
    z['h4_ret3'] = c.pct_change(48)
    z['h4_vol_ratio'] = v.rolling(16).mean() / v.rolling(80).mean().replace(0, np.nan)
    z['h4_range_pct'] = (z['high'].rolling(16).max() - z['low'].rolling(16).min()) / c
    h4 = z.loc[last_idx].set_index('date')[['h4_ret','h4_ret3','h4_vol_ratio','h4_range_pct']]
    snap = snap.join(h4, how='left', rsuffix='_h4')
    d = daily.copy(); d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    df = daily_features(d); wf = weekly_features(d)
    snap = snap.join(df, how='left')
    # Attach latest completed weekly feature as of each snapshot without future leakage.
    wf2 = wf.reset_index().rename(columns={wf.index.name or 'index':'week_date'}).sort_values('week_date')
    s2 = snap.reset_index().rename(columns={'date':'snapshot_date'}).sort_values('snapshot_date')
    s2 = pd.merge_asof(s2, wf2, left_on='snapshot_date', right_on='week_date', direction='backward')
    s2 = s2.set_index('snapshot_date')
    # Benchmark daily context and outcomes.
    sp = spy_daily.copy(); sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize()
    s2['spy_close'] = sp['close'].reindex(s2.index)
    s2['spy_day_ret5'] = sp['close'].pct_change(5).reindex(s2.index)
    s2['spy_day_ret20'] = sp['close'].pct_change(20).reindex(s2.index)
    sd = spy_15.copy()
    if not sd.empty:
        sd['date'] = sd['datetime'].dt.normalize(); sd['spy_h4_ret'] = sd['close'].pct_change(16)
        slast = sd.loc[sd.groupby('date')['datetime'].idxmax()].set_index('date')
        s2['spy_h4_ret'] = slast['spy_h4_ret'].reindex(s2.index)
        s2['rel_h4_vs_spy'] = s2['h4_ret'] - s2['spy_h4_ret']
    stock_close = d['close']
    for h in [1,5,10]:
        s2[f'fwd_{h}d_return'] = stock_close.shift(-h).reindex(s2.index) / stock_close.reindex(s2.index) - 1
        s2[f'spy_fwd_{h}d_return'] = sp['close'].shift(-h).reindex(s2.index) / sp['close'].reindex(s2.index) - 1
        s2[f'fwd_{h}d_excess'] = s2[f'fwd_{h}d_return'] - s2[f'spy_fwd_{h}d_return']
    s2['symbol'] = symbol
    s2['close'] = d['close'].reindex(s2.index)
    keep = [c for c in s2.columns if c.startswith(('m15_','h4_','day_','week_','spy_','rel_','fwd_'))] + ['symbol','close']
    s2 = s2[keep]
    return s2.reset_index()


def main():
    symbols = select_symbols()
    fetch_symbols = symbols + ([] if 'SPY' in symbols else ['SPY'])
    print('pilot symbols:', ', '.join(symbols))
    daily = daily_history(fetch_symbols)
    intraday = {}
    failures = {}
    for s in fetch_symbols:
        try:
            intraday[s] = fetch_15m(s)
            if intraday[s].empty:
                failures[s] = 'no 15m history'
        except Exception as e:
            failures[s] = type(e).__name__
            intraday[s] = pd.DataFrame()
    spy_d = daily.get('SPY', pd.DataFrame())
    spy_15 = intraday.get('SPY', pd.DataFrame())
    parts = []
    coverage = {}
    for s in symbols:
        try:
            part = build_symbol(s, intraday.get(s, pd.DataFrame()), daily.get(s, pd.DataFrame()), spy_d, spy_15)
            if not part.empty:
                parts.append(part)
                coverage[s] = {'rows': len(part), 'first': str(part['snapshot_date'].min().date()), 'last': str(part['snapshot_date'].max().date())}
            else:
                failures[s] = failures.get(s, 'no training rows')
        except Exception as e:
            failures[s] = type(e).__name__ + ': ' + str(e)[:120]
    train = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not train.empty:
        train.to_csv(OUT, index=False, compression='gzip')
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'status': 'PILOT POINT-IN-TIME TRAINING DATASET — NOT A FORECAST',
        'history_source_15m': 'Twelve Data',
        'framework': '15m + rolling 4h + Day + Week + SPY context',
        'start_target': START_DATE,
        'symbols_requested': len(symbols),
        'symbols_with_training_rows': len(coverage),
        'training_rows': int(len(train)),
        'first_snapshot': str(train['snapshot_date'].min()) if not train.empty else None,
        'last_snapshot': str(train['snapshot_date'].max()) if not train.empty else None,
        'outcomes': ['1-session return/excess vs SPY','5-session return/excess vs SPY','10-session return/excess vs SPY'],
        'lookahead_control': 'All input features are computed at or before each snapshot date; future returns are stored only as outcome columns.',
        'coverage': coverage,
        'failures': failures,
    }
    SUMMARY.write_text(json.dumps(summary, separators=(',', ':')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
