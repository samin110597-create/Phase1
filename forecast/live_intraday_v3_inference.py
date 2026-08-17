from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL = Path('forecast/data/intraday_signal_v3.joblib')
SUMMARY = Path('docs/data/intraday_training_summary.json')
_BUNDLE = None

NON_LIVE_FEATURES = {
    'candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank','decision_price_proxy'
}


def _activation_check():
    if not MODEL.exists() or not SUMMARY.exists():
        return False, 'model or training summary missing'
    try:
        summary=json.loads(SUMMARY.read_text())
        if not summary.get('benchmark_label_alignment','').startswith('CORRECTED:'):
            return False, 'same-snapshot SPY label correction not recorded'
        bundle=joblib.load(MODEL)
        cols=set(bundle.get('feature_columns',[]))
        bad=sorted(cols & NON_LIVE_FEATURES)
        if bad:
            return False, 'non-live-reproducible model features present: ' + ','.join(bad)
        if bundle.get('status') != 'FROZEN_FOR_PROSPECTIVE_VALIDATION':
            return False, 'model is not frozen for prospective validation'
        return True, 'activation checks passed'
    except Exception as e:
        return False, f'activation check failed: {type(e).__name__}'


def available():
    ok,_=_activation_check()
    return ok


def activation_status():
    ok,reason=_activation_check()
    return {'ready':ok,'reason':reason}


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        ok,reason=_activation_check()
        if not ok:
            raise RuntimeError('V3 activation blocked: '+reason)
        _BUNDLE = joblib.load(MODEL)
    return _BUNDLE


def _calibrated(cal, raw_p):
    p=np.clip(np.asarray(raw_p,dtype=float),1e-5,1-1e-5)
    x=np.log(p/(1-p)).reshape(-1,1)
    return cal.predict_proba(x)[:,1]


def predict(feature_values: dict):
    ok,reason=_activation_check()
    if not ok:
        return {'status':'MODEL_V3_BUILDING_OR_BLOCKED','activation_reason':reason,'horizons':{},'validated_horizons':[],'model_version':'intraday-signal-v3'}
    b=_bundle(); cols=b['feature_columns']; x=pd.DataFrame([{c:feature_values.get(c,np.nan) for c in cols}])
    out={}
    for h,sides in b.get('models',{}).items():
        row={}
        for side,item in sides.items():
            m=item['model']; cal=item['calibrator']; raw=m.predict_proba(x[cols])[:,1]
            p=float(_calibrated(cal,raw)[0])
            row[f'p_{side}']=round(p,4)
            row[f'{side}_threshold']=float(item.get('threshold',0.60))
            row[f'{side}_selection_model']=item.get('selection_model')
            row[f'{side}_threshold_passed']=bool(p >= float(item.get('threshold',0.60)))
        pu=row.get('p_up'); pdn=row.get('p_down')
        if pu is not None and pdn is not None and pu+pdn <= 1:
            row['p_neutral']=round(1-pu-pdn,4)
        else:
            row['p_neutral']=None
        row['accepted_for_display']=False
        row['display_side']=None
        out[str(h)]=row
    return {
        'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION',
        'horizons':out,
        'validated_horizons':[],
        'model_version':b.get('version','intraday-signal-v3'),
        'trained_at':b.get('trained_at'),
        'feature_count':len(cols),
        'truth_note':'This model is trained on live-like intraday snapshots and frozen. It is not labeled validated until prospective outcomes accumulate without retuning.'
    }
