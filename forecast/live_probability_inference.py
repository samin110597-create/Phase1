from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path('forecast/data/live_probability_model.joblib')


def _d(v, scale=1.0):
    try:
        x = float(v)
        return x * scale if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _b(v):
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    return np.nan


def current_features(m15, h4, day, week, quote, spy15, minute_of_day):
    m15 = m15 or {}; h4 = h4 or {}; day = day or {}; week = week or {}; quote = quote or {}; spy15 = spy15 or {}
    spy10 = _d(spy15.get('return_10bars_pct'), 0.01)
    m10 = _d(m15.get('return_10bars_pct'), 0.01)
    prev = _d(quote.get('previous_close')); price = _d(quote.get('price'))
    session_change = price / prev - 1 if np.isfinite(price) and np.isfinite(prev) and prev else np.nan
    row = {
        'm15_ret3': _d(m15.get('return_3bars_pct'),0.01),
        'm15_ret10': m10,
        'm15_rsi14': _d(m15.get('rsi14')),
        'm15_above_ema20': _b(m15.get('above_ema20')),
        'm15_ema20_gt_ema50': _b(m15.get('ema20_above_ema50')),
        'm15_macd_delta_pct': _d(m15.get('macd_delta_pct'),0.01),
        'm15_atr_pct': _d(m15.get('atr_pct'),0.01),
        'm15_vol_ratio20': _d(m15.get('volume_ratio20')),
        'm15_vwap20_dist': _d(m15.get('rolling_vwap20_distance_pct'),0.01),
        'h4_ret': _d(h4.get('return_4h_pct'),0.01),
        'h4_ret3': _d(h4.get('return_12h_pct'),0.01),
        'h4_vol_ratio': _d(h4.get('volume_ratio')),
        'h4_range_pct': _d(h4.get('range_pct'),0.01),
        'day_ret3': _d(day.get('return_3bars_pct'),0.01),
        'day_ret10': _d(day.get('return_10bars_pct'),0.01),
        'day_rsi14': _d(day.get('rsi14')),
        'day_above_ema20': _b(day.get('above_ema20')),
        'day_ema20_gt_ema50': _b(day.get('ema20_above_ema50')),
        'day_macd_delta_pct': _d(day.get('macd_delta_pct'),0.01),
        'day_atr_pct': _d(day.get('atr_pct'),0.01),
        'day_vol_ratio20': _d(day.get('volume_ratio20')),
        'day_vwap20_dist': _d(day.get('rolling_vwap20_distance_pct'),0.01),
        'week_ret3': _d(week.get('return_3bars_pct'),0.01),
        'week_ret10': _d(week.get('return_10bars_pct'),0.01),
        'week_rsi14': _d(week.get('rsi14')),
        'week_above_ema20': _b(week.get('above_ema20')),
        'week_ema20_gt_ema50': _b(week.get('ema20_above_ema50')),
        'week_macd_delta_pct': _d(week.get('macd_delta_pct'),0.01),
        'week_atr_pct': _d(week.get('atr_pct'),0.01),
        'week_vol_ratio20': _d(week.get('volume_ratio20')),
        'week_vwap20_dist': _d(week.get('rolling_vwap20_distance_pct'),0.01),
        'spy_m15_ret10': spy10,
        'rel_m15_vs_spy': m10 - spy10 if np.isfinite(m10) and np.isfinite(spy10) else np.nan,
        'session_change': session_change,
        'minute_norm': (float(minute_of_day) - 570.0) / 390.0,
    }
    return row


def calibrated_probabilities(m15, h4, day, week, quote, spy15, minute_of_day):
    if not MODEL_PATH.exists():
        return {'status':'MODEL_NOT_TRAINED','horizons':{}}
    artifact = joblib.load(MODEL_PATH)
    row = current_features(m15,h4,day,week,quote,spy15,minute_of_day)
    results = {}
    for h, bundle in artifact.get('horizons', {}).items():
        features = bundle['features']
        X = pd.DataFrame([[row.get(c, np.nan) for c in features]], columns=features)
        raw = float(bundle['model'].predict_proba(X)[:,1][0])
        raw = min(max(raw,1e-5),1-1e-5)
        logit = np.array([[math.log(raw/(1-raw))]])
        up = float(bundle['calibrator'].predict_proba(logit)[:,1][0])
        metrics = bundle.get('metrics',{})
        accepted = bool(metrics.get('accepted_for_display'))
        results[str(h)] = {
            'p_up': round(up,4),
            'p_down': round(1-up,4),
            'accepted_for_display': accepted,
            'validation': {
                'holdout_n': metrics.get('holdout_n'),
                'brier': metrics.get('brier'),
                'base_brier': metrics.get('base_brier'),
                'log_loss': metrics.get('log_loss'),
                'base_log_loss': metrics.get('base_log_loss'),
                'ece_10bin': metrics.get('ece_10bin'),
            }
        }
    accepted_any = any(x.get('accepted_for_display') for x in results.values())
    return {'status':'VALIDATED_V1' if accepted_any else 'CALIBRATED_NOT_VALIDATED','horizons':results,
            'model_version':artifact.get('version'),'trained_at':artifact.get('trained_at')}
