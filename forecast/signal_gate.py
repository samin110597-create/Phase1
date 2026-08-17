from __future__ import annotations

import json
from pathlib import Path

BACKTEST = Path('docs/data/ranked_candidate_validation.json')


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


def _tf_direction(row: dict):
    out = {}
    m15 = row.get('m15') or {}
    h4 = row.get('h4') or {}
    day = row.get('day') or {}
    week = row.get('week') or {}

    if m15:
        bull = bool(m15.get('above_ema20')) and (_num(m15.get('macd_delta_pct')) or 0) >= 0
        bear = (not bool(m15.get('above_ema20'))) and (_num(m15.get('macd_delta_pct')) or 0) < 0
        out['15m'] = 'UP' if bull else 'DOWN' if bear else 'NEUTRAL'
    if h4 and _num(h4.get('return_4h_pct')) is not None:
        out['4h'] = 'UP' if float(h4['return_4h_pct']) > 0 else 'DOWN' if float(h4['return_4h_pct']) < 0 else 'NEUTRAL'
    if day:
        bull = bool(day.get('above_ema20')) and bool(day.get('ema20_above_ema50'))
        bear = (not bool(day.get('above_ema20'))) and (not bool(day.get('ema20_above_ema50')))
        out['Day'] = 'UP' if bull else 'DOWN' if bear else 'NEUTRAL'
    if week:
        bull = bool(week.get('above_ema20')) and bool(week.get('ema20_above_ema50'))
        bear = (not bool(week.get('above_ema20'))) and (not bool(week.get('ema20_above_ema50')))
        out['Week'] = 'UP' if bull else 'DOWN' if bear else 'NEUTRAL'
    return out


def _backtest_metrics(horizon: str, side: str):
    try:
        data = json.loads(BACKTEST.read_text())
        return data.get('metrics', {}).get(str(horizon), {}).get(side.lower(), {})
    except Exception:
        return {}


def _historical_edge(metrics: dict):
    auc = _num(metrics.get('roc_auc'))
    brier = _num(metrics.get('brier'))
    base_brier = _num(metrics.get('base_brier'))
    ll = _num(metrics.get('log_loss'))
    base_ll = _num(metrics.get('base_log_loss'))
    return bool(
        auc is not None and auc >= 0.55
        and brier is not None and base_brier is not None and brier < base_brier
        and ll is not None and base_ll is not None and ll < base_ll
    )


def evaluate_signal(row: dict, market_open: bool):
    """Create a conservative research signal candidate from live probability + MTF + backtest evidence.

    This does not turn an unvalidated probability into a validated one. It is an abstention gate:
    most observations should return NO_SIGNAL.
    """
    tf = _tf_direction(row)
    probs = (row.get('probability_model') or {}).get('horizons', {})
    quote = row.get('quote') or {}
    quote_age = _num(quote.get('age_seconds'))
    fresh = (not market_open) or (quote_age is not None and quote_age <= 180)

    candidates = []
    for h in ('5', '10'):
        p = probs.get(h) or {}
        for side, key in (('UP', 'p_up'), ('DOWN', 'p_down')):
            prob = _num(p.get(key))
            if prob is None:
                continue
            metrics = _backtest_metrics(h, side)
            hist = _historical_edge(metrics)
            aligned = sum(1 for v in tf.values() if v == side)
            opposed = sum(1 for v in tf.values() if v not in (side, 'NEUTRAL'))
            dayweek_ok = all(tf.get(k, side) in (side, 'NEUTRAL') for k in ('Day', 'Week'))

            strict_validated = bool(p.get('accepted_for_display')) and p.get('display_side') == side
            if strict_validated and fresh and aligned >= 3 and dayweek_ok:
                status = 'STRICT_VALIDATED_SIGNAL_CANDIDATE'
            elif hist and fresh and prob >= 0.60 and aligned >= 3 and opposed <= 1 and dayweek_ok:
                status = 'EVIDENCE_ALIGNED_SIGNAL_CANDIDATE'
            elif hist and fresh and prob >= 0.56 and aligned >= 3 and opposed <= 1:
                status = 'WATCHLIST'
            else:
                status = 'NO_SIGNAL'

            candidates.append({
                'horizon_sessions': int(h),
                'side': side,
                'probability': round(prob, 4),
                'status': status,
                'fresh_quote': fresh,
                'mtf_alignment': aligned,
                'mtf_opposed': opposed,
                'timeframes': tf,
                'historical_edge_supported': hist,
                'historical_auc': metrics.get('roc_auc'),
                'historical_brier': metrics.get('brier'),
                'historical_base_brier': metrics.get('base_brier'),
                'historical_log_loss': metrics.get('log_loss'),
                'historical_base_log_loss': metrics.get('base_log_loss'),
            })

    priority = {
        'STRICT_VALIDATED_SIGNAL_CANDIDATE': 4,
        'EVIDENCE_ALIGNED_SIGNAL_CANDIDATE': 3,
        'WATCHLIST': 2,
        'NO_SIGNAL': 1,
    }
    candidates.sort(key=lambda x: (priority[x['status']], x['probability'], x['mtf_alignment']), reverse=True)
    best = candidates[0] if candidates else {
        'status': 'NO_SIGNAL', 'side': None, 'horizon_sessions': None, 'probability': None,
        'fresh_quote': fresh, 'mtf_alignment': 0, 'timeframes': tf,
        'historical_edge_supported': False,
    }
    return {
        'status': best['status'],
        'best': best,
        'all_gates': candidates,
        'policy': 'abstain unless probability, MTF structure, data freshness and historical evidence agree',
    }
