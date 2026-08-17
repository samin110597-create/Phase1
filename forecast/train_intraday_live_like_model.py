from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('forecast/data/intraday_live_like_v2.csv.gz')
MODEL = Path('forecast/data/intraday_signal_v3.joblib')
VALIDATION = Path('docs/data/intraday_model_v3_validation.json')
MODEL.parent.mkdir(parents=True, exist_ok=True)
VALIDATION.parent.mkdir(parents=True, exist_ok=True)

ID_COLS = {'snapshot_dt','slot','symbol','close','split_like_gap'}
DROP_PREFIX = ('fwd_','spy_fwd_','label_','move_threshold_')


def feature_columns(df: pd.DataFrame):
    cols=[]
    for c in df.columns:
        if c in ID_COLS or c.startswith(DROP_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def model_factory(name: str):
    if name == 'logistic':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(C=0.35, max_iter=800, class_weight='balanced', solver='lbfgs')),
        ])
    if name == 'hgb2':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=2.0, min_samples_leaf=80, random_state=7)),
        ])
    if name == 'hgb4':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingClassifier(max_iter=260, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=4.0, min_samples_leaf=120, random_state=17)),
        ])
    raise ValueError(name)


def fit_calibrator(raw_p: np.ndarray, y: np.ndarray):
    p=np.clip(raw_p,1e-5,1-1e-5)
    x=np.log(p/(1-p)).reshape(-1,1)
    cal=LogisticRegression(C=1.0,max_iter=500)
    cal.fit(x,y)
    return cal


def calibrated(cal, raw_p):
    p=np.clip(np.asarray(raw_p),1e-5,1-1e-5)
    x=np.log(p/(1-p)).reshape(-1,1)
    return cal.predict_proba(x)[:,1]


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return {'n':int(len(y)),'base_rate':float(np.mean(y)) if len(y) else None}
    base=float(np.mean(y)); bp=np.full(len(y),base)
    return {
        'n':int(len(y)), 'base_rate':base,
        'brier':float(brier_score_loss(y,p)), 'base_brier':float(brier_score_loss(y,bp)),
        'log_loss':float(log_loss(y,np.c_[1-p,p],labels=[0,1])), 'base_log_loss':float(log_loss(y,np.c_[1-bp,bp],labels=[0,1])),
        'roc_auc':float(roc_auc_score(y,p)),
    }


def precision_k(frame: pd.DataFrame, p: np.ndarray, label: str, k: int):
    z=frame[['snapshot_dt',label]].copy(); z['p']=p
    picks=z.sort_values(['snapshot_dt','p'],ascending=[True,False]).groupby('snapshot_dt').head(k)
    return {'k':k,'n':int(len(picks)),'snapshots':int(picks['snapshot_dt'].nunique()),'precision':float(picks[label].mean()) if len(picks) else None,'mean_probability':float(picks['p'].mean()) if len(picks) else None}


def threshold_test(frame: pd.DataFrame, p: np.ndarray, label: str):
    z=frame[['snapshot_dt',label]].copy(); z['p']=p
    best=None
    for t in np.arange(0.52,0.76,0.01):
        eligible=z[z['p']>=t]
        if eligible.empty: continue
        picks=eligible.sort_values(['snapshot_dt','p'],ascending=[True,False]).groupby('snapshot_dt').head(1)
        if len(picks)<40: continue
        pr=float(picks[label].mean()); avg=float(picks['p'].mean()); gap=abs(pr-avg)
        score=pr - 0.35*gap + min(len(picks),200)/10000
        cand={'threshold':round(float(t),2),'signals':int(len(picks)),'precision':pr,'mean_probability':avg,'calibration_gap':gap,'score':score}
        if best is None or cand['score']>best['score']: best=cand
    if best: best.pop('score',None)
    return best


def choose_pool(df: pd.DataFrame, side: str):
    if side=='up': return df[df['candidate_rank_pct']>=0.75].copy()
    return df[df['candidate_rank_pct']<=0.25].copy()


def main():
    if not DATA.exists(): raise RuntimeError('intraday live-like dataset not built yet')
    df=pd.read_csv(DATA,compression='gzip',parse_dates=['snapshot_dt'])
    features=feature_columns(df)
    results={}; bundle={'version':'intraday-signal-v3','feature_columns':features,'models':{},'trained_at':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION'}

    # 2026 has already been inspected in earlier development, so it is diagnostic only here.
    periods={
        'train':df['snapshot_dt']<pd.Timestamp('2024-01-01'),
        'calibration':(df['snapshot_dt']>=pd.Timestamp('2024-01-01'))&(df['snapshot_dt']<pd.Timestamp('2025-01-01')),
        'selection':(df['snapshot_dt']>=pd.Timestamp('2025-01-01'))&(df['snapshot_dt']<pd.Timestamp('2026-01-01')),
        'development_2026':df['snapshot_dt']>=pd.Timestamp('2026-01-01'),
    }

    for h in (1,5,10):
        results[str(h)]={}; bundle['models'][str(h)]={}
        for side in ('up','down'):
            label=f'label_{side}_{h}'
            pool=choose_pool(df,side)
            pool=pool[pool[label].notna()].copy(); pool[label]=pool[label].astype(int)
            splits={name:pool.loc[mask.reindex(pool.index,fill_value=False)] for name,mask in periods.items()}
            if min(len(splits['train']),len(splits['calibration']),len(splits['selection']))<250:
                results[str(h)][side]={'status':'INSUFFICIENT_HISTORY','counts':{k:int(len(v)) for k,v in splits.items()}}; continue

            trials=[]; chosen=None
            for name in ('logistic','hgb2','hgb4'):
                m=model_factory(name); m.fit(splits['train'][features],splits['train'][label])
                raw_cal=m.predict_proba(splits['calibration'][features])[:,1]
                cal=fit_calibrator(raw_cal,splits['calibration'][label].to_numpy())
                p_sel=calibrated(cal,m.predict_proba(splits['selection'][features])[:,1])
                met=metrics(splits['selection'][label].to_numpy(),p_sel)
                p1=precision_k(splits['selection'],p_sel,label,1); p3=precision_k(splits['selection'],p_sel,label,3)
                th=threshold_test(splits['selection'],p_sel,label)
                # Optimize selective precision first, then probability quality.
                score=(p1['precision'] or 0)*2 + (p3['precision'] or 0) - met.get('brier',1) - 0.5*met.get('log_loss',1)
                trial={'model':name,'selection_metrics':met,'precision_at_1':p1,'precision_at_3':p3,'threshold':th,'score':score,'model_obj':m,'cal_obj':cal}
                trials.append(trial)
                if chosen is None or score>chosen['score']: chosen=trial

            name=chosen['model']; m=chosen['model_obj']; cal=chosen['cal_obj']
            dev=splits['development_2026']; p_dev=calibrated(cal,m.predict_proba(dev[features])[:,1]) if len(dev) else np.array([])
            dev_metrics=metrics(dev[label].to_numpy(),p_dev) if len(dev) else {'n':0}
            dev_p1=precision_k(dev,p_dev,label,1) if len(dev) else {'k':1,'n':0}; dev_p3=precision_k(dev,p_dev,label,3) if len(dev) else {'k':3,'n':0}

            # Refit selected family for forward use. Keep the newest ~20% as a calibration block.
            full=pool.sort_values('snapshot_dt'); cut=max(250,int(len(full)*0.80)); fit=full.iloc[:cut]; final_cal=full.iloc[cut:]
            fm=model_factory(name); fm.fit(fit[features],fit[label]); fcal=fit_calibrator(fm.predict_proba(final_cal[features])[:,1],final_cal[label].to_numpy())
            threshold=chosen['threshold']['threshold'] if chosen.get('threshold') else 0.60

            bundle['models'][str(h)][side]={'model':fm,'calibrator':fcal,'threshold':threshold,'selection_model':name}
            clean_trials=[]
            for t in trials:
                clean_trials.append({k:v for k,v in t.items() if k not in ('model_obj','cal_obj','score')})
            results[str(h)][side]={
                'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION','selected_model':name,'feature_count':len(features),
                'train_n':int(len(splits['train'])),'calibration_n':int(len(splits['calibration'])),'selection_2025_n':int(len(splits['selection'])),'development_2026_n':int(len(dev)),
                'selection_2025':chosen['selection_metrics'],'selection_precision_at_1':chosen['precision_at_1'],'selection_precision_at_3':chosen['precision_at_3'],'frozen_signal_threshold':threshold,
                'development_2026':dev_metrics,'development_2026_precision_at_1':dev_p1,'development_2026_precision_at_3':dev_p3,
                'model_trials':clean_trials,
                'truth_note':'2026 is development diagnostic, not an untouched holdout. The frozen model must earn validation prospectively after deployment.'
            }

    joblib.dump(bundle,MODEL,compress=3)
    summary={
        'generated_at':datetime.now(timezone.utc).isoformat(),'status':'INTRADAY MODEL V3 FROZEN FOR PROSPECTIVE VALIDATION — NOT YET A VALIDATED TRADING SIGNAL',
        'dataset_rows':int(len(df)),'symbols':int(df['symbol'].nunique()),'decision_times':sorted(df['slot'].dropna().unique().tolist()),
        'features':features,'model_file':str(MODEL),'validation_design':'pre-2024 fit; 2024 calibration; 2025 model/threshold selection; 2026 development diagnostic only; final proof is prospective from deployment forward',
        'selection_objective':'Precision@1 and Precision@3 inside historical top-10 bullish/downside candidate pools, with Brier/log-loss penalties',
        'results':results,
        'forward_acceptance_policy':'Do not label VALIDATED until a frozen prospective sample is large enough to evaluate precision, calibration, Brier/log-loss, and regime stability without retuning on those outcomes.'
    }
    VALIDATION.write_text(json.dumps(summary,separators=(',',':')))
    print(json.dumps({'status':summary['status'],'rows':summary['dataset_rows'],'symbols':summary['symbols'],'results':{h:{s:{'model':v.get('selected_model'),'threshold':v.get('frozen_signal_threshold'),'p1_2025':(v.get('selection_precision_at_1') or {}).get('precision'),'p1_2026_dev':(v.get('development_2026_precision_at_1') or {}).get('precision')} for s,v in sides.items()} for h,sides in results.items()}},indent=2))

if __name__=='__main__':
    main()
