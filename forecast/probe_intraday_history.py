from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

OUT = Path('docs/data/history_probe.json')
TIMEOUT = 15
SYMBOL = os.getenv('PHASE1_HISTORY_PROBE_SYMBOL', 'AAPL')
YEARS = [2026, 2024, 2022, 2020, 2016, 2010]


def get_json(url: str):
    req = Request(url, headers={'User-Agent': 'Phase1-Forecast-Research/1.0'})
    try:
        with urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')[:500]
        return {'_http_error': e.code, '_body': body}, e.code
    except Exception as e:
        return {'_error': type(e).__name__}, None


def summarize_rows(rows, ts_key=None, date_key=None):
    if not isinstance(rows, list) or not rows:
        return {'ok': False, 'count': 0}
    dates = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if ts_key and r.get(ts_key):
            try:
                dates.append(datetime.fromtimestamp(float(r[ts_key]) / 1000, tz=timezone.utc).isoformat())
            except Exception:
                pass
        elif date_key and r.get(date_key):
            dates.append(str(r[date_key]))
    return {
        'ok': True,
        'count': len(rows),
        'oldest': min(dates) if dates else None,
        'newest': max(dates) if dates else None,
    }


def twelve_probe(key: str, year: int):
    q = urlencode({
        'symbol': SYMBOL,
        'interval': '15min',
        'start_date': f'{year}-01-04',
        'end_date': f'{year}-01-11',
        'outputsize': 5000,
        'apikey': key,
    })
    d, status = get_json('https://api.twelvedata.com/time_series?' + q)
    if isinstance(d, dict) and isinstance(d.get('values'), list):
        out = summarize_rows(d['values'], date_key='datetime')
        out['http_status'] = status
        return out
    return {'ok': False, 'count': 0, 'http_status': status, 'message': str(d.get('message') or d.get('_body') or d.get('_error') or 'no values')[:220]}


def polygon_probe(key: str, year: int):
    q = urlencode({'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': key})
    url = f'https://api.polygon.io/v2/aggs/ticker/{quote(SYMBOL)}/range/15/minute/{year}-01-04/{year}-01-11?' + q
    d, status = get_json(url)
    rows = d.get('results') if isinstance(d, dict) else None
    if isinstance(rows, list) and rows:
        out = summarize_rows(rows, ts_key='t')
        out['http_status'] = status
        out['provider_status'] = d.get('status')
        return out
    return {'ok': False, 'count': 0, 'http_status': status, 'provider_status': d.get('status') if isinstance(d, dict) else None, 'message': str((d or {}).get('error') or (d or {}).get('message') or (d or {}).get('_body') or 'no results')[:220]}


def fmp_probe(key: str, year: int):
    q = urlencode({'symbol': SYMBOL, 'from': f'{year}-01-04', 'to': f'{year}-01-11', 'apikey': key})
    d, status = get_json('https://financialmodelingprep.com/stable/historical-chart/15min?' + q)
    if isinstance(d, list) and d:
        out = summarize_rows(d, date_key='date')
        out['http_status'] = status
        return out
    return {'ok': False, 'count': 0, 'http_status': status, 'message': str((d or {}).get('Error Message') if isinstance(d, dict) else 'no results')[:220]}


def alpha_probe(key: str, year: int):
    q = urlencode({
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': SYMBOL,
        'interval': '15min',
        'month': f'{year}-01',
        'outputsize': 'full',
        'adjusted': 'true',
        'extended_hours': 'false',
        'apikey': key,
    })
    d, status = get_json('https://www.alphavantage.co/query?' + q)
    if isinstance(d, dict):
        series = next((v for k, v in d.items() if k.startswith('Time Series') and isinstance(v, dict)), None)
        if series:
            keys = list(series.keys())
            return {'ok': True, 'count': len(keys), 'oldest': min(keys), 'newest': max(keys), 'http_status': status}
        msg = d.get('Information') or d.get('Note') or d.get('Error Message') or d.get('_body') or 'no series'
        return {'ok': False, 'count': 0, 'http_status': status, 'message': str(msg)[:220]}
    return {'ok': False, 'count': 0, 'http_status': status, 'message': 'unexpected response'}


def main():
    providers = {
        'twelve_data': (os.getenv('TWELVE_DATA_API_KEY'), twelve_probe),
        'polygon_massive': (os.getenv('POLYGON_API_KEY'), polygon_probe),
        'fmp': (os.getenv('FMP_API_KEY'), fmp_probe),
        'alpha_vantage': (os.getenv('ALPHA_VANTAGE_API_KEY'), alpha_probe),
    }
    results = {}
    for name, (key, fn) in providers.items():
        if not key:
            results[name] = {'configured': False, 'years': {}}
            continue
        years = {}
        for year in YEARS:
            years[str(year)] = fn(key, year)
        successful = [int(y) for y, r in years.items() if r.get('ok')]
        results[name] = {
            'configured': True,
            'years': years,
            'oldest_successful_probe_year': min(successful) if successful else None,
            'newest_successful_probe_year': max(successful) if successful else None,
        }
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'status': 'HISTORICAL INTRADAY COVERAGE PROBE — NO FORECAST OUTPUT',
        'symbol': SYMBOL,
        'interval': '15m',
        'probe_windows': [f'{y}-01-04 to {y}-01-11' for y in YEARS],
        'providers': results,
        'purpose': 'Choose the deepest reliable provider before building Forecast V1 historical multi-timeframe training data.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print(json.dumps({k: {'configured': v['configured'], 'oldest_successful_probe_year': v.get('oldest_successful_probe_year')} for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
