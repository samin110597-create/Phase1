from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path('forecast/data/live_probability_model.joblib')


def _d(v, scale=1.0):
    try:
        x=float(v); return x*scale if math.isfinite(x) else np.nan
    except Exception: return np.nan


def _b(v):
    if v is True: return 1.0
    if v is False: return 0.0
    return np.nan


def current_features(m15,h4,day,week,quote,spy15,minute_of_day):
    m15=m15 or {}; h4=h4 or {}; day=day or {}; week=week or {}; quote=quote or {}; spy15=spy15 or {}
    spy10=_d(spy15.get('return_10bars_pct'),0.01); m10=_d(m15.get('return_10bars_pct'),0.01)
    prev=_d(quote.get('previous_close')); price=_d(quote.get('price')); session_change=price/prev-1 if np.isfinite(price) and np.isfinite(prev) and prev else np.nan
    return {
      'm15_ret3':_d(m15.get('return_3bars_pct'),0.01),'m15_ret10':m10,'m15_rsi14':_d(m15.get('rsi14')),
      'm15_above_ema20':_b(m15.get('above_ema20')),'m15_ema20_gt_ema50':_b(m15.get('ema20_above_ema50')),
      'm15_macd_delta_pct':_d(m15.get('macd_delta_pct'),0.01),'m15_atr_pct':_d(m15.get('atr_pct'),0.01),
      'm15_vol_ratio20':_d(m15.get('volume_ratio20')),'m15_vwap20_dist':_d(m15.get('rolling_vwap20_distance_pct'),0.01),
      'h4_ret':_d(h4.get('return_4h_pct'),0.01),'h4_ret3':_d(h4.get('return_12h_pct'),0.01),'h4_vol_ratio':_d(h4.get('volume_ratio')),'h4_range_pct':_d(h4.get('range_pct'),0.01),
      'day_ret3':_d(day.get('return_3bars_pct'),0.01),'day_ret10':_d(day.get('return_10bars_pct'),0.01),'day_rsi14':_d(day.get('rsi14')),
      'day_above_ema20':_b(day.get('above_ema20')),'day_ema20_gt_ema50':_b(day.get('ema20_above_ema50')),
      'day_macd_delta_pct':_d(day.get('macd_delta_pct'),0.01),'day_atr_pct':_d(day.get('atr_pct'),0.01),'day_vol_ratio20':_d(day.get('volume_ratio20')),'day_vwap20_dist':_d(day.get('rolling_vwap20_distance_pct'),0.01),
      'week_ret3':_d(week.get('return_3bars_pct'),0.01),'week_ret10':_d(week.get('return_10bars_pct'),0.01),'week_rsi14':_d(week.get('rsi14')),
      'week_above_ema20':_b(week.get('above_ema20')),'week_ema20_gt_ema50':_b(week.get('ema20_above_ema50')),
      'week_macd_delta_pct':_d(week.get('macd_delta_pct'),0.01),'week_atr_pct':_d(week.get('atr_pct'),0.01),'week_vol_ratio20':_d(week.get('volume_ratio20')),'week_vwap20_dist':_d(week.get('rolling_vwap20_distance_pct'),0.01),
      'spy_m15_ret10':spy10,'rel_m15_vs_spy':m10-spy10 if np.isfinite(m10) and np.isfinite(spy10) else np.nan,
      'session_change':session_change,'minute_norm':(float(minute_of_day)-570.0)/390.0,
    }


def calibrated_probabilities(m15,h4,day,week,quote,spy15,minute_of_day):
    if not MODEL_PATH.exists(): return {'status':'MODEL_NOT_TRAINED','horizons':{}}
    artifact=joblib.load(MODEL_PATH); row=current_features(m15,h4,day,week,quote,spy15,minute_of_day); results={}
    for h,bundle in artifact.get('horizons',{}).items():
        features=bundle['features']; X=pd.DataFrame([[row.get(c,np.nan) for c in features]],columns=features)
        raw=float(bundle['model'].predict_proba(X)[:,1][0]); raw=min(max(raw,1e-5),1-1e-5); logit=np.array([[math.log(raw/(1-raw))]])
        up=float(bundle['calibrator'].predict_proba(logit)[:,1][0]); down=1-up; metrics=bundle.get('metrics',{}); sides=set(metrics.get('accepted_sides',[])); thresholds=bundle.get('display_thresholds',{})
        up_thr=thresholds.get('up'); down_thr=thresholds.get('down'); display_side=None
        if 'up' in sides and up_thr is not None and up>=float(up_thr): display_side='UP'
        if 'down' in sides and down_thr is not None and down>=float(down_thr):
            if display_side is None or down>up: display_side='DOWN'
        results[str(h)]={
          'p_up':round(up,4),'p_down':round(down,4),'accepted_for_display':display_side is not None,'display_side':display_side,
          'accepted_sides':sorted(sides),'display_thresholds':thresholds,
          'validation':{'final_2026_n':metrics.get('final_2026_n'),'brier':metrics.get('brier'),'base_brier':metrics.get('base_brier'),'log_loss':metrics.get('log_loss'),'base_log_loss':metrics.get('base_log_loss'),'ece_10bin':metrics.get('ece_10bin')}
        }
    available=[h for h,p in results.items() if p.get('accepted_for_display')]
    validated=[str(h) for h,b in artifact.get('horizons',{}).items() if b.get('metrics',{}).get('accepted_for_display')]
    return {'status':'STRICT_VALIDATED_V2' if validated else 'NO_HORIZON_PASSED_STRICT_VALIDATION','horizons':results,'current_display_horizons':available,'validated_horizons':validated,'model_version':artifact.get('version'),'trained_at':artifact.get('trained_at')}
