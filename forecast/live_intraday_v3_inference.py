from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL = Path('forecast/data/intraday_signal_v3.joblib')
_BUNDLE = None


def available():
    return MODEL.exists()


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL)
    return _BUNDLE


def _calibrated(cal, raw_p):
    p=np.clip(np.asarray(raw_p,dtype=float),1e-5,1-1e-5)
    x=np.log(p/(1-p)).reshape(-1,1)
    return cal.predict_proba(x)[:,1]


def predict(feature_values: dict):
    if not MODEL.exists():
        return {'status':'MODEL_V3_BUILDING','horizons':{},'validated_horizons':[],'model_version':'intraday-signal-v3'}
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
        # V3 is frozen for prospective validation. Threshold crossing is a research signal candidate,
        # not historical proof of a validated trading signal.
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
