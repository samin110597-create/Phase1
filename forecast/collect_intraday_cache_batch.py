from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import forecast.build_intraday_live_like_dataset as base


def main():
    raw = os.getenv('PHASE1_BATCH_SYMBOLS', '')
    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]
    if not symbols:
        raise RuntimeError('PHASE1_BATCH_SYMBOLS is empty')

    results = {}
    for symbol in symbols:
        try:
            bars = base.fetch_15m(symbol)
            results[symbol] = {
                'rows': int(len(bars)),
                'first': str(bars['datetime'].min()) if len(bars) else None,
                'last': str(bars['datetime'].max()) if len(bars) else None,
                'ok': bool(len(bars)),
            }
        except Exception as exc:
            results[symbol] = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:180]}'}

    failed = [s for s, r in results.items() if not r.get('ok')]
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'symbols': symbols,
        'results': results,
        'failed': failed,
    }
    Path('forecast/cache/intraday15').mkdir(parents=True, exist_ok=True)
    Path('forecast/cache/intraday15/batch_report.json').write_text(json.dumps(report, separators=(',', ':')))
    print(json.dumps(report, indent=2))
    if failed:
        raise RuntimeError('historical cache collection failed for: ' + ','.join(failed))


if __name__ == '__main__':
    main()
