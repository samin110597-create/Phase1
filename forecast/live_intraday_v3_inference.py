from __future__ import annotations

import json, math
from pathlib import Path
import joblib, numpy as np, pandas as pd

MODEL=Path('forecast/data/intraday_signal_v3.joblib'); SUMMARY=Path('docs/data/intraday_training_summary.json'); AUDIT=Path('docs/data/intraday_dataset_audit.json')
_BUNDLE=None
NON_LIVE_FEATURES={'decision_price_proxy'}


def _activation_check():
    if not MODEL.exists() or not SUMMARY.exists() or not AUDIT.exists(): return False,'model, training summary, or dataset audit missing'
    try:
        summary=json.loads(SUMMARY.read_text()); audit=json.loads(AUDIT.read_text())
        if audit.get('status')!='PASS': return False,'dataset integrity audit did not pass'
        if not summary.get('benchmark_label_alignment','').startswith('CORRECTED:'): return False,'same-snapshot SPY label correction not recorded'
        b=joblib.load(MODEL); bad=sorted(set(b.get('feature_columns',[]))&NON_LIVE_FEATURES)
        if bad:return False,'non-live-reproducible model features present: '+','.join(bad)
        if b.get('status')!='FROZEN_FOR_PROSPECTIVE_VALIDATION':return False,'model is not frozen for prospective validation'
        return True,'activation checks passed'
    except Exception as e:return False,f'activation check failed: {type(e).__name__}: {str(e)[:100]}'

def available():return _activation_check()[0]
def activation_status():
    ok,reason=_activation_check(); return {'ready':ok,'reason':reason}

def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        ok,reason=_activation_check()
        if not ok: raise RuntimeError('model activation blocked: '+reason)
        _BUNDLE=joblib.load(MODEL)
    return _BUNDLE

def _platt(cal,raw):
    p=np.clip(np.asarray(raw,float),1e-5,1-1e-5); x=np.log(p/(1-p)).reshape(-1,1); return cal.predict_proba(x)[:,1]

def _regime_key(v):
    def bit(k):
        x=v.get(k); return '1' if x is not None and pd.notna(x) and float(x)>.5 else '0'
    return '|'.join([bit('spy_above_ema20'),bit('vix_above_ema20'),bit('sector_above_ema20'),str(v.get('slot','NA'))])

def _pbin(p):return int(max(0,min(9,math.floor(float(p)*10))))

def _meta_predict(b,feature_values):
    cols=b['feature_columns']; X=pd.DataFrame([{c:feature_values.get(c,np.nan) for c in cols}]); out={}
    policy=b.get('meta_policy',{}); min_n=int(policy.get('min_local_n',60)); prior=float(policy.get('prior_strength',40)); max_disp=float(policy.get('max_member_dispersion',.09)); reg=_regime_key(feature_values)
    for h,item in b.get('models',{}).items():
        member_ps=[]
        for mem in item.get('members',[]):
            m=mem['model']; raw=m.predict_proba(X[cols]); cls=list(m.classes_); cp=np.zeros(3)
            for k in (0,1,2): cp[k]=float(_platt(mem['calibrators'][str(k)],raw[:,cls.index(k)])[0])
            cp=cp/cp.sum() if cp.sum()>0 else np.array([1/3]*3); member_ps.append(cp)
        if not member_ps: continue
        stack=np.stack(member_ps); ens=stack.mean(axis=0); disp=float(np.max(stack.std(axis=0))); adj=np.zeros(3); local=[]
        for k in (0,1,2):
            bb=_pbin(ens[k]); z=item.get('local_table',{}).get(f'{reg}|{k}|{bb}')
            if not z or int(z.get('n',0))<min_n:z=item.get('global_table',{}).get(f'{k}|{bb}')
            n=int(z.get('n',0)) if z else 0; rate=float(z.get('rate',ens[k])) if z else float(ens[k]); adj[k]=(n*rate+prior*ens[k])/(n+prior) if n else ens[k]; local.append(n)
        adj=adj/adj.sum() if adj.sum()>0 else np.array([1/3]*3)
        thresholds=item.get('thresholds',{}); rs=item.get('return_stats',{})
        row={'p_down':round(float(adj[0]),4),'p_neutral':round(float(adj[1]),4),'p_up':round(float(adj[2]),4),'model_dispersion':round(disp,4),'comparable_setup_n':{'down':local[0],'neutral':local[1],'up':local[2]},'up_threshold':thresholds.get('up'),'down_threshold':thresholds.get('down'),'up_threshold_passed':bool(thresholds.get('up') is not None and adj[2]>=float(thresholds['up']) and local[2]>=min_n and disp<=max_disp),'down_threshold_passed':bool(thresholds.get('down') is not None and adj[0]>=float(thresholds['down']) and local[0]>=min_n and disp<=max_disp),'historical_return_targets':rs,'accepted_for_display':False,'display_side':None}
        out[str(h)]=row
    return {'status':'FROZEN_META_MODEL_AWAITING_FORWARD_VALIDATION','horizons':out,'validated_horizons':[],'model_version':b.get('version','intraday-meta-v6'),'trained_at':b.get('trained_at'),'feature_count':len(cols),'truth_note':'Probability blends a calibrated model ensemble with observed rates from comparable historical regime/probability buckets. Final validation remains prospective.'}

def _legacy_predict(b,feature_values):
    cols=b['feature_columns']; x=pd.DataFrame([{c:feature_values.get(c,np.nan) for c in cols}]); out={}
    for h,sides in b.get('models',{}).items():
        row={}
        for side,item in sides.items():
            raw=item['model'].predict_proba(x[cols])[:,1]; p=float(_platt(item['calibrator'],raw)[0]); row[f'p_{side}']=round(p,4); row[f'{side}_threshold']=float(item.get('threshold',.60)); row[f'{side}_threshold_passed']=bool(p>=float(item.get('threshold',.60)))
        pu=row.get('p_up'); pdn=row.get('p_down'); row['p_neutral']=round(1-pu-pdn,4) if pu is not None and pdn is not None and pu+pdn<=1 else None; row['accepted_for_display']=False; row['display_side']=None; out[str(h)]=row
    return {'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION','horizons':out,'validated_horizons':[],'model_version':b.get('version','intraday-signal-v3'),'trained_at':b.get('trained_at'),'feature_count':len(cols)}

def predict(feature_values:dict):
    ok,reason=_activation_check()
    if not ok:return {'status':'MODEL_BUILDING_OR_BLOCKED','activation_reason':reason,'horizons':{},'validated_horizons':[],'model_version':'pending'}
    b=_bundle(); return _meta_predict(b,feature_values) if b.get('version') in ('intraday-meta-v5','intraday-meta-v6') else _legacy_predict(b,feature_values)
