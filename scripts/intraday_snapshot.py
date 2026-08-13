from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

LATEST = Path('docs/data/latest.json')
OUT = Path('docs/data/intraday.json')
MAX_SYMBOLS = int(os.getenv('PHASE1_LIVE_SYMBOLS', '8'))
TIMEOUT = 8


def get_json(url: str, headers: dict | None = None):
    req = Request(url, headers=headers or {'User-Agent': 'Phase1-Momentum/1.0'})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode('utf-8'))


def as_float(v):
    try:
        x = float(v)
        return x if x > 0 else None
    except Exception:
        return None


def as_epoch(v, scale: float = 1.0):
    try:
        x = float(v) / scale
        return int(x) if x > 0 else None
    except Exception:
        return None


def twelve_quote(symbol: str, key: str):
    p = urlencode({'symbol': symbol, 'apikey': key})
    d = get_json('https://api.twelvedata.com/quote?' + p)
    price = as_float(d.get('extended_price')) if d.get('is_market_open') is False and d.get('extended_price') else as_float(d.get('close'))
    if d.get('status') == 'error' or not price:
        return None
    ts = as_epoch(d.get('extended_timestamp')) if d.get('extended_price') and d.get('extended_timestamp') else as_epoch(d.get('last_quote_at') or d.get('timestamp'))
    return {'provider': 'Twelve Data', 'price': price, 'timestamp': ts, 'exchange': d.get('exchange'), 'market_open': d.get('is_market_open')}


def finnhub_quote(symbol: str, key: str):
    p = urlencode({'symbol': symbol, 'token': key})
    d = get_json('https://finnhub.io/api/v1/quote?' + p)
    price = as_float(d.get('c'))
    if not price:
        return None
    return {'provider': 'Finnhub', 'price': price, 'timestamp': as_epoch(d.get('t'))}


def fmp_quote(symbol: str, key: str):
    p = urlencode({'symbol': symbol, 'apikey': key})
    d = get_json('https://financialmodelingprep.com/stable/quote?' + p)
    row = d[0] if isinstance(d, list) and d else d if isinstance(d, dict) else {}
    price = as_float(row.get('price'))
    if not price:
        return None
    return {'provider': 'FMP', 'price': price, 'timestamp': as_epoch(row.get('timestamp')), 'exchange': row.get('exchange')}


def polygon_quote(symbol: str, key: str):
    p = urlencode({'apiKey': key})
    d = get_json(f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{quote(symbol)}?' + p)
    t = d.get('ticker') or {}
    trade = t.get('lastTrade') or {}
    minute = t.get('min') or {}
    price = as_float(trade.get('p')) or as_float(minute.get('c')) or as_float((t.get('day') or {}).get('c'))
    if not price:
        return None
    # Polygon trade timestamps are nanoseconds; minute timestamps are milliseconds.
    ts = as_epoch(trade.get('t'), 1_000_000_000) or as_epoch(t.get('updated'), 1_000_000_000) or as_epoch(minute.get('t'), 1_000)
    return {'provider': 'Polygon', 'price': price, 'timestamp': ts}


def alpha_vantage_quote(symbol: str, key: str):
    p = urlencode({'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': key})
    d = get_json('https://www.alphavantage.co/query?' + p)
    row = d.get('Global Quote') or {}
    price = as_float(row.get('05. price'))
    if not price:
        return None
    # GLOBAL_QUOTE often exposes a trading date rather than an intraday epoch.
    day = row.get('07. latest trading day')
    ts = None
    if day:
        try:
            ts = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass
    return {'provider': 'Alpha Vantage', 'price': price, 'timestamp': ts, 'note': 'Freshness depends on Alpha Vantage market-data entitlement'}


def fetch_all(symbol: str, keys: dict):
    calls = [
        ('TWELVE_DATA_API_KEY', twelve_quote),
        ('FINNHUB_API_KEY', finnhub_quote),
        ('FMP_API_KEY', fmp_quote),
        ('POLYGON_API_KEY', polygon_quote),
        ('ALPHA_VANTAGE_API_KEY', alpha_vantage_quote),
    ]
    out, errors = [], []
    for env_name, fn in calls:
        key = keys.get(env_name)
        if not key:
            continue
        try:
            q = fn(symbol, key)
            if q:
                out.append(q)
            else:
                errors.append({'provider': env_name.replace('_API_KEY', ''), 'error': 'no usable quote returned'})
        except Exception as e:
            errors.append({'provider': env_name.replace('_API_KEY', ''), 'error': type(e).__name__})
    return out, errors


def consensus(symbol: str, quotes: list[dict], errors: list[dict]):
    if not quotes:
        return {'symbol': symbol, 'sources': [], 'errors': errors}

    stamped = [q for q in quotes if q.get('timestamp')]
    newest = max((q['timestamp'] for q in stamped), default=None)
    # Keep quotes within 30 minutes of the freshest timestamp. This prevents a stale
    # EOD provider from contaminating a current intraday consensus.
    fresh = [q for q in quotes if newest is None or (q.get('timestamp') and q['timestamp'] >= newest - 1800)]
    if not fresh:
        fresh = quotes

    prices = [q['price'] for q in fresh]
    med = statistics.median(prices)
    spread_pct = ((max(prices) - min(prices)) / med * 100) if len(prices) > 1 and med else 0.0
    within_35bp = sum(1 for p in prices if abs(p / med - 1) <= 0.0035)
    agreement_pct = within_35bp / len(prices) * 100 if prices else 0.0

    # Prefer Twelve Data for the display quote when it is part of the fresh cohort;
    # otherwise use the freshest available provider. Consensus is reported separately.
    primary = next((q for q in fresh if q['provider'] == 'Twelve Data'), None)
    if not primary:
        primary = max(fresh, key=lambda q: q.get('timestamp') or 0)

    return {
        'symbol': symbol,
        'price': round(primary['price'], 6),
        'consensus_price': round(med, 6),
        'quote_timestamp': primary.get('timestamp'),
        'freshest_timestamp': newest,
        'primary_provider': primary['provider'],
        'providers_used': len(fresh),
        'providers_returned': len(quotes),
        'agreement_pct': round(agreement_pct, 1),
        'provider_spread_pct': round(spread_pct, 4),
        'sources': quotes,
        'errors': errors,
    }


def main():
    keys = {name: os.getenv(name) for name in [
        'TWELVE_DATA_API_KEY', 'FINNHUB_API_KEY', 'FMP_API_KEY', 'POLYGON_API_KEY', 'ALPHA_VANTAGE_API_KEY'
    ]}
    configured = [name for name, value in keys.items() if value]
    if not configured:
        raise RuntimeError('No market-data API secrets are configured')

    latest = json.loads(LATEST.read_text())
    stocks = sorted(latest.get('stocks', []), key=lambda x: (x.get('confluence', 0), x.get('confidence', 0)), reverse=True)
    symbols = [x['symbol'] for x in stocks[:MAX_SYMBOLS]]
    rows = []
    for symbol in symbols:
        quotes, errors = fetch_all(symbol, keys)
        rows.append(consensus(symbol, quotes, errors))

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'GitHub Actions near-live snapshot; not tick-by-tick streaming',
        'configured_providers': [x.replace('_API_KEY', '').replace('_', ' ') for x in configured],
        'symbols_requested': len(symbols),
        'symbols_with_quotes': sum(1 for x in rows if x.get('price')),
        'agreement_rule': 'Fresh sources within 30 minutes of newest quote; agreement = price within 0.35% of fresh-source median',
        'quotes': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print('configured providers:', ', '.join(payload['configured_providers']))
    print('wrote', payload['symbols_with_quotes'], 'multi-source quote rows')


if __name__ == '__main__':
    main()
