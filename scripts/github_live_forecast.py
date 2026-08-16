from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path('docs/data/live_forecast.json')
TIMEOUT = 10
NY = ZoneInfo('America/New_York')
FINNHUB_KEY = os.getenv('FINNHUB_API_KEY')
TWELVE_KEY = os.getenv('TWELVE_DATA_API_KEY')

# Broad liquid research universe. SPY is benchmark-only and is added separately.
UNIVERSE = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AMD','AVGO',
    'NFLX','JPM','BAC','XOM','CVX','WMT','COST','HD','UNH','LLY','ORCL','CRM',
    'PLTR','MU','INTC','QCOM','AMAT','TSM','UBER','V','MA','GS','MS','CAT','GE',
    'DIS','KO','PEP','CSCO','ADBE'
]


def now_utc():
    return datetime.now(timezone.utc)


def market_state():
    n = datetime.now(NY)
    minute = n.hour * 60 + n.minute
    is_weekday = n.weekday() < 5
    is_open = is_weekday and 570 <= minute < 960
    return {
        'market_open': is_open,
        'ny_time': n.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'session': 'REGULAR' if is_open else 'CLOSED_OR_OFF_HOURS'
    }


def get_json(url: str):
    req = Request(url, headers={'User-Agent': 'Phase1-GitHub-Live/1.0'})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode('utf-8'))


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def finnhub_quote(symbol: str):
    if not FINNHUB_KEY:
        raise RuntimeError('FINNHUB_API_KEY missing')
    u = 'https://finnhub.io/api/v1/quote?' + urlencode({'symbol': symbol, 'token': FINNHUB_KEY})
    d = get_json(u)
    price = safe_float(d.get('c'))
    if not price or price <= 0:
        raise RuntimeError('no usable Finnhub quote')
    ts = int(d.get('t') or 0) or None
    return {
        'price': price,
        'previous_close': safe_float(d.get('pc')),
        'day_open': safe_float(d.get('o')),
        'day_high': safe_float(d.get('h')),
        'day_low': safe_float(d.get('l')),
        'timestamp': ts,
        'age_seconds': max(0, int(time.time() - ts)) if ts else None,
        'provider': 'Finnhub'
    }


def twelve_15m(symbol: str, outputsize: int = 220):
    if not TWELVE_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY missing')
    u = 'https://api.twelvedata.com/time_series?' + urlencode({
        'symbol': symbol,
        'interval': '15min',
        'outputsize': outputsize,
        'timezone': 'America/New_York',
        'apikey': TWELVE_KEY,
    })
    d = get_json(u)
    vals = d.get('values') if isinstance(d, dict) else None
    if not isinstance(vals, list) or not vals:
        raise RuntimeError((d.get('message') if isinstance(d, dict) else None) or 'no Twelve Data 15m bars')
    x = pd.DataFrame(vals)
    x['datetime'] = pd.to_datetime(x['datetime'], errors='coerce')
    for c in ['open','high','low','close','volume']:
        x[c] = pd.to_numeric(x.get(c), errors='coerce')
    x = x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime')
    return x[['datetime','open','high','low','close','volume']].reset_index(drop=True)


def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14):
    prev = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev).abs(),
        (df['low'] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def frame_features(df: pd.DataFrame, label: str):
    if df is None or len(df) < 55:
        return None
    x = df.copy()
    c = x['close'].astype(float)
    v = x['volume'].fillna(0).astype(float)
    e12, e26, e20, e50 = ema(c,12), ema(c,26), ema(c,20), ema(c,50)
    macd = e12 - e26
    sig = ema(macd,9)
    a = atr(x,14)
    typical = (x['high'] + x['low'] + x['close']) / 3
    vol20 = v.rolling(20).mean()
    vw20 = (typical*v).rolling(20).sum() / v.rolling(20).sum().replace(0,np.nan)
    last = len(x)-1
    def pctchg(n):
        if len(c) <= n or c.iloc[-n-1] == 0:
            return None
        return (c.iloc[-1] / c.iloc[-n-1] - 1) * 100
    return {
        'label': label,
        'last_bar': str(x['datetime'].iloc[last]) if 'datetime' in x.columns else str(x.index[last]),
        'close': round(float(c.iloc[-1]), 6),
        'return_3bars_pct': round(pctchg(3), 3) if pctchg(3) is not None else None,
        'return_10bars_pct': round(pctchg(10), 3) if pctchg(10) is not None else None,
        'rsi14': round(float(rsi(c,14).iloc[-1]), 2) if pd.notna(rsi(c,14).iloc[-1]) else None,
        'above_ema20': bool(c.iloc[-1] > e20.iloc[-1]),
        'ema20_above_ema50': bool(e20.iloc[-1] > e50.iloc[-1]),
        'macd_delta_pct': round(float((macd.iloc[-1]-sig.iloc[-1]) / c.iloc[-1] * 100), 4) if c.iloc[-1] else None,
        'atr_pct': round(float(a.iloc[-1] / c.iloc[-1] * 100), 3) if pd.notna(a.iloc[-1]) and c.iloc[-1] else None,
        'volume_ratio20': round(float(v.iloc[-1] / vol20.iloc[-1]), 3) if pd.notna(vol20.iloc[-1]) and vol20.iloc[-1] else None,
        'rolling_vwap20_distance_pct': round(float((c.iloc[-1] / vw20.iloc[-1] - 1) * 100), 3) if pd.notna(vw20.iloc[-1]) and vw20.iloc[-1] else None,
    }


def rolling_4h(df15: pd.DataFrame):
    if df15 is None or len(df15) < 80:
        return None
    c = df15['close'].astype(float)
    v = df15['volume'].fillna(0).astype(float)
    hi = df15['high'].rolling(16).max().iloc[-1]
    lo = df15['low'].rolling(16).min().iloc[-1]
    v16 = v.rolling(16).mean().iloc[-1]
    v80 = v.rolling(80).mean().iloc[-1]
    return {
        'label': 'rolling 4h',
        'last_bar': str(df15['datetime'].iloc[-1]),
        'return_4h_pct': round(float((c.iloc[-1]/c.iloc[-17]-1)*100),3),
        'return_12h_pct': round(float((c.iloc[-1]/c.iloc[-49]-1)*100),3),
        'range_pct': round(float((hi-lo)/c.iloc[-1]*100),3) if c.iloc[-1] else None,
        'volume_ratio': round(float(v16/v80),3) if v80 else None,
    }


def daily_download(symbols):
    raw = yf.download(symbols, period='18mo', auto_adjust=True, actions=False, group_by='ticker', threads=True, progress=False)
    out = {}
    for s in symbols:
        try:
            x = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            x = x.rename(columns=str.lower).dropna(subset=['close'])
            x.index = pd.to_datetime(x.index).tz_localize(None)
            x = x.reset_index().rename(columns={x.index.name or 'index':'datetime','Date':'datetime','date':'datetime'})
            if 'datetime' not in x.columns:
                x = x.rename(columns={x.columns[0]:'datetime'})
            out[s] = x[['datetime','open','high','low','close','volume']].copy()
        except Exception:
            out[s] = pd.DataFrame()
    return out


def weekly_from_daily(d: pd.DataFrame):
    if d is None or d.empty:
        return pd.DataFrame()
    x = d.set_index(pd.to_datetime(d['datetime'])).copy()
    w = pd.DataFrame({
        'open': x['open'].resample('W-FRI').first(),
        'high': x['high'].resample('W-FRI').max(),
        'low': x['low'].resample('W-FRI').min(),
        'close': x['close'].resample('W-FRI').last(),
        'volume': x['volume'].resample('W-FRI').sum(),
    }).dropna(subset=['close']).reset_index().rename(columns={'index':'datetime'})
    return w


def daily_direction_score(day, week, quote, spy_day):
    s = 0.0
    if day:
        s += 10 if day['above_ema20'] else -10
        s += 10 if day['ema20_above_ema50'] else -10
        if day.get('rsi14') is not None:
            s += 6 if day['rsi14'] >= 55 else -6 if day['rsi14'] <= 45 else 0
        if day.get('macd_delta_pct') is not None:
            s += 7 if day['macd_delta_pct'] >= 0 else -7
        if day.get('return_10bars_pct') is not None and spy_day and spy_day.get('return_10bars_pct') is not None:
            s += 8 if day['return_10bars_pct'] >= spy_day['return_10bars_pct'] else -8
    if week:
        s += 7 if week['above_ema20'] else -7
        s += 6 if week['ema20_above_ema50'] else -6
    if quote and quote.get('previous_close'):
        chg = (quote['price']/quote['previous_close']-1)*100
        s += max(-10,min(10,chg*3))
    return s


def final_score(day, week, m15, h4, quote, spy15):
    s = 0.0
    reasons = []
    def add(cond, pts, pos, neg):
        nonlocal s
        if cond is True:
            s += pts; reasons.append(pos)
        elif cond is False:
            s -= pts; reasons.append(neg)
    if m15:
        add(m15['above_ema20'], 9, '15m above EMA20', '15m below EMA20')
        add(m15['ema20_above_ema50'], 8, '15m EMA trend positive', '15m EMA trend negative')
        if m15.get('rsi14') is not None:
            if 55 <= m15['rsi14'] <= 75: s += 6; reasons.append('15m RSI supportive')
            elif m15['rsi14'] <= 45: s -= 6; reasons.append('15m RSI weak')
        if m15.get('macd_delta_pct') is not None:
            s += 6 if m15['macd_delta_pct'] >= 0 else -6
            reasons.append('15m MACD positive' if m15['macd_delta_pct'] >= 0 else '15m MACD negative')
        if m15.get('rolling_vwap20_distance_pct') is not None:
            s += 6 if m15['rolling_vwap20_distance_pct'] >= 0 else -6
            reasons.append('above rolling VWAP' if m15['rolling_vwap20_distance_pct'] >= 0 else 'below rolling VWAP')
        if spy15 and m15.get('return_10bars_pct') is not None and spy15.get('return_10bars_pct') is not None:
            rel = m15['return_10bars_pct'] - spy15['return_10bars_pct']
            s += 8 if rel >= 0 else -8
            reasons.append('beating SPY intraday' if rel >= 0 else 'lagging SPY intraday')
    if h4 and h4.get('return_4h_pct') is not None:
        s += 9 if h4['return_4h_pct'] >= 0 else -9
        reasons.append('4h momentum positive' if h4['return_4h_pct'] >= 0 else '4h momentum negative')
    if day:
        add(day['above_ema20'], 7, 'daily above EMA20', 'daily below EMA20')
        add(day['ema20_above_ema50'], 6, 'daily trend positive', 'daily trend negative')
    if week:
        add(week['above_ema20'], 5, 'weekly above EMA20', 'weekly below EMA20')
        add(week['ema20_above_ema50'], 4, 'weekly trend positive', 'weekly trend negative')
    if quote and quote.get('previous_close'):
        live_chg = (quote['price']/quote['previous_close']-1)*100
        adj = max(-8,min(8,live_chg*2))
        s += adj
        reasons.append(f'live session change {live_chg:+.2f}%')
    up = max(0,min(100,50+s))
    down = max(0,min(100,50-s))
    bias = 'BULLISH' if s >= 14 else 'BEARISH' if s <= -14 else 'NEUTRAL'
    return round(s,2), round(up,1), round(down,1), bias, reasons[:9]


def main():
    if not FINNHUB_KEY or not TWELVE_KEY:
        raise RuntimeError('FINNHUB_API_KEY and TWELVE_DATA_API_KEY must already exist in GitHub Secrets')

    state = market_state()
    daily = daily_download(UNIVERSE + ['SPY'])
    spy_day = frame_features(daily.get('SPY'), 'day') if not daily.get('SPY', pd.DataFrame()).empty else None

    quote_rows = {}
    quote_errors = {}
    for s in UNIVERSE:
        try:
            quote_rows[s] = finnhub_quote(s)
        except Exception as e:
            quote_errors[s] = type(e).__name__
        time.sleep(0.05)

    pre = []
    day_cache, week_cache = {}, {}
    for s in UNIVERSE:
        d = daily.get(s, pd.DataFrame())
        if d.empty or s not in quote_rows:
            continue
        day = frame_features(d, 'day')
        week = frame_features(weekly_from_daily(d), 'week')
        if not day or not week:
            continue
        day_cache[s], week_cache[s] = day, week
        q = quote_rows[s]
        adv20 = float((d['close'] * d['volume']).tail(20).mean()) if len(d) >= 20 else 0
        if q['price'] < 5 or adv20 < 100_000_000:
            continue
        ds = daily_direction_score(day, week, q, spy_day)
        activity = abs((q['price']/q['previous_close']-1)*100) if q.get('previous_close') else 0
        pre.append({'symbol':s,'direction_score':ds,'activity':activity,'avg_dollar_volume_20d':adv20})

    bullish = sorted(pre, key=lambda x:x['direction_score'], reverse=True)[:3]
    bearish = sorted(pre, key=lambda x:x['direction_score'])[:3]
    used = {x['symbol'] for x in bullish + bearish}
    wildcard = next((x for x in sorted(pre,key=lambda x:x['activity'],reverse=True) if x['symbol'] not in used), None)
    shortlist = bullish + bearish + ([wildcard] if wildcard else [])
    shortlist_symbols = list(dict.fromkeys(x['symbol'] for x in shortlist))[:7]

    intraday = {}
    intraday_errors = {}
    for s in shortlist_symbols + ['SPY']:
        try:
            intraday[s] = twelve_15m(s)
        except Exception as e:
            intraday_errors[s] = str(e)[:120]
        # Current Twelve Data Basic budget behaves as ~8 credits/minute.
        time.sleep(0.15)

    spy15 = frame_features(intraday.get('SPY'), '15m') if 'SPY' in intraday else None
    rows = []
    for s in shortlist_symbols:
        if s not in intraday:
            rows.append({'symbol':s,'error':intraday_errors.get(s,'no intraday data')})
            continue
        m15 = frame_features(intraday[s], '15m')
        h4 = rolling_4h(intraday[s])
        day = day_cache.get(s)
        week = week_cache.get(s)
        q = quote_rows.get(s)
        edge, up, down, bias, reasons = final_score(day, week, m15, h4, q, spy15)
        stale = bool(state['market_open'] and q and q.get('age_seconds') is not None and q['age_seconds'] > 180)
        rows.append({
            'symbol': s,
            'quote': q,
            'm15': m15,
            'h4': h4,
            'day': day,
            'week': week,
            'signed_edge': edge,
            'upside_score': up,
            'downside_score': down,
            'bias': bias,
            'reasons': reasons,
            'live_stale': stale,
            'forecast_status': 'MARKET_CLOSED' if not state['market_open'] else 'STALE_QUOTE' if stale else 'PROVISIONAL_NEAR_LIVE',
        })

    valid = [r for r in rows if not r.get('error') and not r.get('live_stale')]
    top_up = sorted(valid, key=lambda x:x['upside_score'], reverse=True)[:3]
    top_down = sorted(valid, key=lambda x:x['downside_score'], reverse=True)[:3]

    payload = {
        'generated_at': now_utc().isoformat(),
        'engine': 'Phase1 GitHub-Only Live Forecast Engine V1',
        'mode': 'GitHub Actions near-live snapshot; refresh target every 5 minutes during U.S. regular market hours',
        'status': 'PROVISIONAL DIRECTIONAL SCORE — NOT A CALIBRATED PROBABILITY',
        'market': state,
        'sources': {
            'live_quote': 'Finnhub quote endpoint',
            'intraday': 'Twelve Data 15-minute bars',
            'day_week': 'Yahoo Finance via yfinance',
            'benchmark': 'SPY'
        },
        'universe_size': len(UNIVERSE),
        'funnel': {
            'quotes_received': len(quote_rows),
            'tradeable_after_liquidity_filter': len(pre),
            'intraday_shortlist': shortlist_symbols,
            'displayed_max_each_direction': 3
        },
        'freshness_rule': 'During regular hours, a Finnhub quote older than 180 seconds is marked stale and excluded from top candidates.',
        'top_upside': [r['symbol'] for r in top_up],
        'top_downside': [r['symbol'] for r in top_down],
        'rows': rows,
        'errors': {'quote_count': len(quote_errors), 'intraday': intraday_errors},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print(json.dumps({
        'generated_at': payload['generated_at'],
        'market': state,
        'quotes_received': len(quote_rows),
        'shortlist': shortlist_symbols,
        'top_upside': payload['top_upside'],
        'top_downside': payload['top_downside'],
    }, indent=2))


if __name__ == '__main__':
    main()
