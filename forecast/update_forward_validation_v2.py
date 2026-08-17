from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import forecast.update_forward_validation as old

LIVE = Path('docs/data/live_forecast.json')
LOG = Path('docs/data/forward_predictions.json')
SUMMARY = Path('docs/data/forward_validation.json')
NY = ZoneInfo('America/New_York')
HORIZONS = (1, 5, 10)
SLOT_WINDOW_MINUTES = 35
FORWARD_TARGET_N = 200
SIGNAL_TARGET_N = 50


def robust_slot(dt: datetime):
    """Assign the first scheduled decision slot up to 35 minutes after it.

    GitHub scheduled jobs can start several minutes late. We keep the intended slot
    fixed and store the actual capture time/lag separately, so timing remains auditable.
    """
    minute = dt.hour * 60 + dt.minute
    for hh, mm in old.SLOTS:
        target = hh * 60 + mm
        lag = minute - target
        if 0 <= lag <= SLOT_WINDOW_MINUTES:
            return f'{hh:02d}:{mm:02d}', lag
    return None, None


def append_snapshot(entries, live, capture_source='LIVE_WORKFLOW'):
    if not live.get('market', {}).get('market_open'):
        return entries
    try:
        dt = datetime.fromisoformat(str(live['generated_at']).replace('Z', '+00:00')).astimezone(NY)
    except Exception:
        return entries
    slot, lag = robust_slot(dt)
    if not slot:
        return entries

    existing = {(x.get('date'), x.get('slot'), x.get('symbol'), x.get('model_version')) for x in entries}
    bench = (live.get('benchmark_snapshot') or {}).get('price')

    for r in live.get('rows', []):
        if r.get('error') or r.get('live_stale') or not (r.get('quote') or {}).get('price'):
            continue
        pm = r.get('probability_model') or {}
        ver = pm.get('model_version') or 'UNKNOWN_MODEL'
        key = (dt.date().isoformat(), slot, r.get('symbol'), ver)
        if key in existing:
            continue
        best = (r.get('signal_engine') or {}).get('best') or {}
        rec = {
            'date': dt.date().isoformat(),
            'slot': slot,
            'actual_capture_time_et': dt.strftime('%H:%M:%S'),
            'capture_lag_minutes': lag,
            'capture_source': capture_source,
            'generated_at': live.get('generated_at'),
            'symbol': r.get('symbol'),
            'snapshot_price': float(r['quote']['price']),
            'benchmark_symbol': 'SPY',
            'benchmark_snapshot_price': float(bench) if bench else None,
            'model_version': ver,
            'model_status': pm.get('status'),
            'signal_status': (r.get('signal_engine') or {}).get('status'),
            'signal_side': best.get('side'),
            'signal_horizon': best.get('horizon_sessions'),
            'horizons': {},
        }
        for h in HORIZONS:
            p = (pm.get('horizons') or {}).get(str(h), {})
            rec['horizons'][str(h)] = {
                'p_up': p.get('p_up'),
                'p_down': p.get('p_down'),
                'p_neutral': p.get('p_neutral'),
                'probability_threshold_up': p.get('up_threshold') or (p.get('display_thresholds') or {}).get('up'),
                'probability_threshold_down': p.get('down_threshold') or (p.get('display_thresholds') or {}).get('down'),
                'move_threshold': p.get('target_move_threshold'),
                'target_definition': p.get('target_definition'),
                'strict_validated': bool(p.get('accepted_for_display')),
                'outcome_known': False,
                'future_close': None,
                'benchmark_future_close': None,
                'stock_return': None,
                'benchmark_return': None,
                'excess_return': None,
                'realized_up': None,
                'realized_down': None,
                'realized_neutral': None,
                'brier_up': None,
                'brier_down': None,
                'signal_success': None,
            }
        entries.append(rec)
        existing.add(key)
    return entries


def backfill_from_git(entries):
    """Recover previously published forecasts from git history without using outcomes.

    Selection depends only on the stored forecast timestamp and fixed decision slots.
    This safely repairs missed logging caused by GitHub cron delays.
    """
    try:
        proc = subprocess.run(
            ['git', 'log', '--since=7 days ago', '--max-count=800', '--format=%H', '--', 'docs/data/live_forecast.json'],
            check=True, capture_output=True, text=True,
        )
        shas = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    except Exception:
        return entries

    snapshots = []
    for sha in shas:
        try:
            show = subprocess.run(
                ['git', 'show', f'{sha}:docs/data/live_forecast.json'],
                check=True, capture_output=True, text=True,
            )
            live = json.loads(show.stdout)
            if not live.get('market', {}).get('market_open'):
                continue
            dt = datetime.fromisoformat(str(live['generated_at']).replace('Z', '+00:00')).astimezone(NY)
            slot, lag = robust_slot(dt)
            if slot:
                snapshots.append((dt, live))
        except Exception:
            continue

    # Oldest qualifying snapshot first means each date/slot/model/symbol keeps the
    # earliest stored forecast after the intended slot, never a cherry-picked later one.
    for _, live in sorted(snapshots, key=lambda z: z[0]):
        entries = append_snapshot(entries, live, capture_source='GIT_HISTORY_BACKFILL')
    return entries


def score_entries(entries):
    symbols = sorted({x.get('symbol') for x in entries if x.get('symbol')} | {'SPY'})
    closes = old.daily_close_history(symbols)
    spy = closes.get('SPY', pd.Series(dtype=float))
    today = pd.Timestamp(datetime.now(NY).date())
    spy_completed = spy[spy.index < today]

    for rec in entries:
        s = rec.get('symbol')
        series = closes.get(s, pd.Series(dtype=float))
        if series.empty:
            continue
        d0 = pd.Timestamp(rec['date'])
        completed = series[series.index < today]
        future_dates = completed.index[completed.index > d0]
        for h in HORIZONS:
            obj = (rec.get('horizons') or {}).get(str(h), {})
            if obj.get('outcome_known') or len(future_dates) < h:
                continue
            fdt = future_dates[h - 1]
            if fdt not in completed.index:
                continue
            fc = float(completed.loc[fdt])
            p_up = obj.get('p_up')
            p_down = obj.get('p_down')
            if p_up is None:
                continue

            # Prefer the same benchmark-relative meaningful-move definition used by
            # the live signal engine whenever the stored snapshot contains it.
            sp0 = rec.get('benchmark_snapshot_price')
            th = obj.get('move_threshold')
            if sp0 is not None and th is not None and fdt in spy_completed.index:
                sret = fc / float(rec['snapshot_price']) - 1
                bfc = float(spy_completed.loc[fdt])
                bret = bfc / float(sp0) - 1
                ex = sret - bret
                if abs(sret) > 0.55:
                    obj['withheld_reason'] = 'split_like_or_extreme_raw_price_discontinuity'
                    continue
                yup = 1 if ex > float(th) else 0
                ydn = 1 if ex < -float(th) else 0
                yn = 1 if not yup and not ydn else 0
                obj.update({
                    'outcome_known': True,
                    'outcome_definition_used': 'MEANINGFUL_EXCESS_RETURN_VS_SPY',
                    'future_date': fdt.date().isoformat(),
                    'future_close': round(fc, 6),
                    'benchmark_future_close': round(bfc, 6),
                    'stock_return': round(sret, 8),
                    'benchmark_return': round(bret, 8),
                    'excess_return': round(ex, 8),
                    'realized_up': yup,
                    'realized_down': ydn,
                    'realized_neutral': yn,
                    'brier_up': round((float(p_up) - yup) ** 2, 8),
                })
                if p_down is not None:
                    obj['brier_down'] = round((float(p_down) - ydn) ** 2, 8)
            else:
                realized = 1 if fc > float(rec['snapshot_price']) else 0
                obj.update({
                    'outcome_known': True,
                    'outcome_definition_used': 'RAW_DIRECTION_FALLBACK',
                    'future_date': fdt.date().isoformat(),
                    'future_close': round(fc, 6),
                    'realized_up': realized,
                    'realized_down': 1 - realized if p_down is not None else None,
                    'brier_up': round((float(p_up) - realized) ** 2, 8),
                })
                if p_down is not None:
                    obj['brier_down'] = round((float(p_down) - (1-realized)) ** 2, 8)

            if rec.get('signal_horizon') == h and rec.get('signal_status') in {
                'FROZEN_FORWARD_SIGNAL_CANDIDATE', 'EVIDENCE_ALIGNED_SIGNAL_CANDIDATE', 'STRICT_VALIDATED_SIGNAL_CANDIDATE'
            }:
                side = rec.get('signal_side')
                obj['signal_success'] = bool(
                    (side == 'UP' and obj.get('realized_up') == 1) or
                    (side == 'DOWN' and obj.get('realized_down') == 1)
                )
    return entries


def wilson_lower(successes, n, z=1.96):
    return old.wilson_lower(successes, n, z)


def side_metrics(rows, side):
    pk = 'p_up' if side == 'up' else 'p_down'
    yk = 'realized_up' if side == 'up' else 'realized_down'
    z = [r for r in rows if r.get(pk) is not None and r.get(yk) is not None]
    if not z:
        return {'n': 0, 'status': 'COLLECTING'}
    p = np.array([float(r[pk]) for r in z])
    y = np.array([int(r[yk]) for r in z])
    base = float(y.mean())
    bp = np.full(len(y), base)
    bins = []
    for lo in np.arange(0, 1, 0.1):
        hi = lo + 0.1
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            bins.append({
                'lo': round(float(lo), 1), 'hi': round(float(hi), 1), 'n': int(m.sum()),
                'mean_probability': round(float(p[m].mean()), 4),
                'observed_rate': round(float(y[m].mean()), 4),
            })
    return {
        'n': int(len(y)),
        'base_rate': round(base, 4),
        'mean_probability': round(float(p.mean()), 4),
        'observed_rate': round(float(y.mean()), 4),
        'brier': round(float(np.mean((p-y)**2)), 6),
        'base_brier': round(float(np.mean((bp-y)**2)), 6),
        'calibration_gap_abs': round(abs(float(p.mean()-y.mean())), 4),
        'calibration_bins': bins,
        'status': 'EARLY_FORWARD_SAMPLE' if len(y) < FORWARD_TARGET_N else 'FORWARD_SAMPLE_AVAILABLE',
    }


def summarize(entries, live):
    active_model = None
    try:
        active_rows = [r for r in live.get('rows', []) if not r.get('error')]
        if active_rows:
            active_model = (active_rows[0].get('probability_model') or {}).get('model_version')
    except Exception:
        pass
    if not active_model and entries:
        active_model = max(entries, key=lambda e: str(e.get('generated_at') or '')).get('model_version')

    active = [e for e in entries if e.get('model_version') == active_model] if active_model else []
    versions = {}
    for e in entries:
        v = e.get('model_version') or 'UNKNOWN_MODEL'
        versions[v] = versions.get(v, 0) + 1

    result = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'status': 'COLLECTING_FORWARD_EVIDENCE',
        'definition': 'Fixed intraday forecasts are stored before outcomes are known. Benchmark-relative meaningful-move scoring is used whenever the snapshot contains SPY and the frozen move threshold.',
        'slots_et': [f'{h:02d}:{m:02d}' for h, m in old.SLOTS],
        'capture_window_minutes': SLOT_WINDOW_MINUTES,
        'forward_target_n_per_horizon': FORWARD_TARGET_N,
        'signal_target_n': SIGNAL_TARGET_N,
        'total_logged_snapshots': len(entries),
        'active_model_version': active_model,
        'active_model_logged_snapshots': len(active),
        'logged_by_model_version': versions,
        'horizons': {},
    }

    for h in HORIZONS:
        matured = []
        sig = []
        for rec in active:
            o = (rec.get('horizons') or {}).get(str(h), {})
            if o.get('outcome_known'):
                matured.append(o)
            if o.get('outcome_known') and o.get('signal_success') is not None:
                sig.append(bool(o['signal_success']))
        up = side_metrics(matured, 'up')
        down = side_metrics(matured, 'down')
        signal_n = len(sig)
        signal_precision = float(np.mean(sig)) if sig else None
        lower = wilson_lower(sum(sig), signal_n) if sig else None
        matured_n = len(matured)
        result['horizons'][str(h)] = {
            'matured_n': matured_n,
            'target_n': FORWARD_TARGET_N,
            'progress_pct': round(min(100.0, 100.0 * matured_n / FORWARD_TARGET_N), 1),
            'up': up,
            'down': down,
            'signal_n': signal_n,
            'signal_target_n': SIGNAL_TARGET_N,
            'signal_precision': round(signal_precision, 4) if signal_precision is not None else None,
            'signal_wilson_lower_95': round(float(lower), 4) if lower is not None else None,
            'status': 'COLLECTING' if matured_n < FORWARD_TARGET_N else 'FORWARD_EVIDENCE_AVAILABLE',
        }
    return result


def main():
    if not LIVE.exists():
        raise RuntimeError('live_forecast.json missing')
    live = old.load_json(LIVE, {})
    entries = old.load_json(LOG, [])
    entries = backfill_from_git(entries)
    entries = append_snapshot(entries, live)
    entries = score_entries(entries)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(entries, separators=(',', ':')))
    summary = summarize(entries, live)
    SUMMARY.write_text(json.dumps(summary, separators=(',', ':')))
    print(json.dumps({
        'logged': len(entries),
        'active_model_version': summary.get('active_model_version'),
        'active_model_logged': summary.get('active_model_logged_snapshots'),
        'summary': str(SUMMARY),
    }, indent=2))


if __name__ == '__main__':
    main()
