from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

LIVE = Path('docs/data/live_forecast.json')
LOG = Path('docs/data/forward_predictions.json')
SUMMARY = Path('docs/data/forward_validation.json')
NY = ZoneInfo('America/New_York')
HORIZONS = (1, 5, 10)
# Keep a small number of fixed, auditable intraday snapshots rather than logging every 5-minute run.
SLOTS = [(10, 0), (12, 0), (14, 0), (15, 45)]
WINDOW_MINUTES = 7


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def near_slot(dt: datetime):
    minute = dt.hour * 60 + dt.minute
    for hh, mm in SLOTS:
        target = hh * 60 + mm
        if 0 <= minute - target < WINDOW_MINUTES:
            return f'{hh:02d}:{mm:02d}'
    return None


def append_current(entries: list[dict], live: dict):
    if not live.get('market', {}).get('market_open'):
        return entries
    dt = datetime.fromisoformat(live['generated_at'].replace('Z', '+00:00')).astimezone(NY)
    slot = near_slot(dt)
    if not slot:
        return entries
    existing = {(x.get('date'), x.get('slot'), x.get('symbol')) for x in entries}
    for r in live.get('rows', []):
        if r.get('error') or r.get('live_stale') or not r.get('quote', {}).get('price'):
            continue
        key = (dt.date().isoformat(), slot, r.get('symbol'))
        if key in existing:
            continue
        hs = r.get('probability_model', {}).get('horizons', {})
        rec = {
            'date': dt.date().isoformat(),
            'slot': slot,
            'generated_at': live.get('generated_at'),
            'symbol': r.get('symbol'),
            'snapshot_price': float(r['quote']['price']),
            'model_version': r.get('probability_model', {}).get('model_version'),
            'model_status': r.get('probability_model', {}).get('status'),
            'horizons': {},
        }
        for h in HORIZONS:
            p = hs.get(str(h), {})
            rec['horizons'][str(h)] = {
                'p_up': p.get('p_up'),
                'p_down': p.get('p_down'),
                'strict_validated': bool(p.get('accepted_for_display')),
                'outcome_known': False,
                'future_close': None,
                'realized_up': None,
                'brier': None,
            }
        entries.append(rec)
        existing.add(key)
    return entries


def daily_close_history(symbols: list[str]):
    if not symbols:
        return {}
    raw = yf.download(symbols, period='2y', auto_adjust=False, actions=False, group_by='ticker', threads=True, progress=False)
    out = {}
    for s in symbols:
        try:
            x = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            close = x['Close'].dropna().copy()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            out[s] = close
        except Exception:
            out[s] = pd.Series(dtype=float)
    return out


def score_entries(entries: list[dict]):
    symbols = sorted({x.get('symbol') for x in entries if x.get('symbol')})
    closes = daily_close_history(symbols)
    today_ny = pd.Timestamp(datetime.now(NY).date())
    for rec in entries:
        s = rec.get('symbol')
        series = closes.get(s, pd.Series(dtype=float))
        if series.empty:
            continue
        d0 = pd.Timestamp(rec['date'])
        # Only completed sessions are eligible. Today's partial daily bar is never used.
        completed = series[series.index < today_ny]
        future_dates = completed.index[completed.index > d0]
        for h in HORIZONS:
            obj = rec.get('horizons', {}).get(str(h), {})
            if obj.get('outcome_known'):
                continue
            if len(future_dates) < h:
                continue
            fdt = future_dates[h - 1]
            future_close = float(completed.loc[fdt])
            p_up = obj.get('p_up')
            if p_up is None:
                continue
            realized_up = 1 if future_close > float(rec['snapshot_price']) else 0
            obj['outcome_known'] = True
            obj['future_date'] = fdt.date().isoformat()
            obj['future_close'] = round(future_close, 6)
            obj['realized_up'] = realized_up
            obj['brier'] = round((float(p_up) - realized_up) ** 2, 8)
    return entries


def wilson_lower(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return None
    p = successes / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    adj = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
    return (center - adj) / den


def summarize(entries: list[dict]):
    result = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'status': 'COLLECTING_FORWARD_EVIDENCE',
        'definition': 'Predictions are logged prospectively at fixed intraday slots; outcome is whether a completed future session close is above the logged snapshot price.',
        'slots_et': [f'{h:02d}:{m:02d}' for h, m in SLOTS],
        'total_logged_snapshots': len(entries),
        'horizons': {},
    }
    for h in HORIZONS:
        rows = []
        for rec in entries:
            obj = rec.get('horizons', {}).get(str(h), {})
            if obj.get('outcome_known') and obj.get('p_up') is not None:
                rows.append((float(obj['p_up']), int(obj['realized_up']), bool(obj.get('strict_validated'))))
        n = len(rows)
        if not n:
            result['horizons'][str(h)] = {'matured_n': 0, 'status': 'COLLECTING'}
            continue
        p = np.array([x[0] for x in rows], dtype=float)
        y = np.array([x[1] for x in rows], dtype=float)
        pred = p >= 0.5
        correct = ((pred & (y == 1)) | (~pred & (y == 0))).astype(int)
        bins = []
        for lo in np.arange(0.0, 1.0, 0.1):
            hi = lo + 0.1
            mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if mask.any():
                bins.append({
                    'lo': round(float(lo), 1), 'hi': round(float(hi), 1), 'n': int(mask.sum()),
                    'mean_probability': round(float(p[mask].mean()), 4),
                    'observed_up_rate': round(float(y[mask].mean()), 4),
                })
        result['horizons'][str(h)] = {
            'matured_n': n,
            'directional_accuracy': round(float(correct.mean()), 4),
            'brier': round(float(np.mean((p-y)**2)), 6),
            'mean_p_up': round(float(p.mean()), 4),
            'observed_up_rate': round(float(y.mean()), 4),
            'wilson_lower_directional_accuracy': round(float(wilson_lower(int(correct.sum()), n)), 4),
            'calibration_bins': bins,
            'status': 'EARLY_FORWARD_SAMPLE' if n < 200 else 'FORWARD_SAMPLE_AVAILABLE',
        }
    return result


def main():
    if not LIVE.exists():
        raise RuntimeError('live_forecast.json missing')
    live = load_json(LIVE, {})
    entries = load_json(LOG, [])
    entries = append_current(entries, live)
    entries = score_entries(entries)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(entries, separators=(',', ':')))
    SUMMARY.write_text(json.dumps(summarize(entries), separators=(',', ':')))
    print(json.dumps({'logged': len(entries), 'summary': str(SUMMARY)}, indent=2))


if __name__ == '__main__':
    main()
