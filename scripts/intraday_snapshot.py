from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

LATEST = Path('docs/data/latest.json')
OUT = Path('docs/data/intraday.json')
MAX_SYMBOLS = 40


def fetch_quote(symbol: str, key: str):
    params = urlencode({'symbol': symbol, 'apikey': key})
    with urlopen('https://api.twelvedata.com/quote?' + params, timeout=8) as r:
        d = json.loads(r.read().decode('utf-8'))
    if d.get('status') == 'error' or 'close' not in d:
        return None
    try:
        price = float(d['close'])
    except Exception:
        return None
    return {
        'symbol': d.get('symbol', symbol),
        'price': price,
        'quote_datetime': d.get('datetime'),
        'quote_timestamp': d.get('timestamp'),
        'exchange': d.get('exchange'),
        'is_extended_hours': bool(d.get('is_extended_hours', False)),
    }


def main():
    key = os.getenv('TWELVE_DATA_API_KEY')
    if not key:
        raise RuntimeError('TWELVE_DATA_API_KEY is not configured')

    latest = json.loads(LATEST.read_text())
    stocks = sorted(latest.get('stocks', []), key=lambda x: (x.get('confluence', 0), x.get('confidence', 0)), reverse=True)
    symbols = [x['symbol'] for x in stocks[:MAX_SYMBOLS]]
    quotes = []
    for s in symbols:
        try:
            q = fetch_quote(s, key)
            if q:
                quotes.append(q)
        except Exception as e:
            print('quote failed', s, e)

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': 'Twelve Data /quote',
        'mode': 'GitHub Actions intraday snapshot; not tick-by-tick streaming',
        'symbols_requested': len(symbols),
        'symbols_returned': len(quotes),
        'quotes': quotes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print('wrote', len(quotes), 'quotes')


if __name__ == '__main__':
    main()
