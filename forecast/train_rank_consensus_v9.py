from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import forecast.train_hourly_selective_v8 as v8
import forecast.train_intraday_meta_model_v5 as meta

STAGE=Path('forecast/data/rank_consensus_v9.joblib')
VAL=Path('docs/data/rank_consensus_v9_validation.json')
STATUS=Path('docs/data/rank_consensus_v9_status.json')
for p in (STAGE,VAL,STATUS): p.parent.mkdir(parents=True,exist_ok=True)

MIN_ALIGN=3
MAX_DISP=.10
MIN_SELECT_N=50
MIN_DEV_N=30


def reg_factory():
    return Pipeline([('impute',SimpleImputer(strategy='median')),('model',HistGradientBoostingRegressor(loss='absolute_error',max_iter=280,learning_rate=.035,max_leaf_nodes=23,min_samples_leaf=100,l2_regularization=5.0,random_state=37))])


def period(frame,start,end=None):
    m=frame.snapshot_dt>=pd.Timestamp(start)
    if end:m &= frame.snapshot_dt<pd.Timestamp(end)
    return frame[m]


def consensus_picks(frame,p,disp,pred_excess,side):
    z=frame[['snapshot_dt','candidate_rank_pct']].copy(); z['p']=p; z['disp']=disp; z['pred_excess']=pred_excess; z['align']=v8.structural_alignment(frame,side)
    tail=z.candidate_rank_pct>=.75 if side=='up' else z.candidate_rank_pct<=.25
    z=z[tail & (z['align']>=MIN_ALIGN) & (z.disp<=MAX_DISP)].copy()
    if z.empty:return z
    z['p_rank']=z.groupby('snapshot_dt').p.rank(pct=True)
    if side=='up': z['ex_rank']=z.groupby('snapshot_dt').pred_excess.rank(pct=True)
    else: z['ex_rank']=z.groupby('snapshot_dt').pred_excess.rank(pct=True,ascending=False)
    z=z[(z.p_rank>=.80)&(z.ex_rank>=.80)].copy()
    if z.empty:return z
    z['score']=.60*z.p_rank+.40*z.ex_rank
    return z.sort_values(['snapshot_dt','score','p'],ascending=[True,False,False]).groupby('snapshot_dt').head(1)


def pick_stats(frame,p,disp,pred_excess,y,side,empirical_probability=None):
    picks=consensus_picks(frame,p,disp,pred_excess,side)
    base=float(np.mean(y)) if len(y) else None
    if picks.empty:return {'signals':0,'precision':None,'base_rate':base,'wilson_lower_95':None,'mean_model_probability':None,'empirical_probability':empirical_probability,'calibration_gap':None},picks
    yy=pd.Series(np.asarray(y,int),index=frame.index); py=yy.loc[picks.index].astype(int); n=len(py); s=int(py.sum()); pr=s/n; lo=v8.wilson(s,n); avg=float(picks.p.mean()); ep=empirical_probability
    gap=abs(pr-ep) if ep is not None else None
    return {'signals':int(n),'precision':float(pr),'base_rate':base,'wilson_lower_95':float(lo),'mean_model_probability':avg,'empirical_probability':ep,'calibration_gap':gap},picks


def empirical_rank_probability(frame,p,disp,pred_excess,y,side):
    stats,_=pick_stats(frame,p,disp,pred_excess,y,side,None)
    n=stats['signals']; pr=stats['precision']; base=stats['base_rate']
    if not n or pr is None:return {'n':0,'rate':base,'shrunk_rate':base}
    prior=40.0
    shr=(n*pr+prior*base)/(n+prior)
    return {'n':int(n),'rate':float(pr),'shrunk_rate':float(shr),'base_rate':float(base)}


def main():
    data,coverage,missing=v8.build_dataset(); feats=v8.features(data)
    pool=data[(data.candidate_rank_pct>=.75)|(data.candidate_rank_pct<=.25)].copy()
    train=pool[pool.snapshot_dt<pd.Timestamp('2025-07-01')]
    platt=period(pool,'2025-07-01','2025-09-01')
    rankcal=period(pool,'2025-09-01','2025-10-01')
    select=period(pool,'2025-10-01','2026-01-01')
    dev=period(pool,'2026-01-01')
    bundle={'version':'rank-consensus-v9','trained_at':datetime.now(timezone.utc).isoformat(),'status':'STAGING','feature_columns':feats,'models':{},'policy':{'min_alignment':MIN_ALIGN,'max_dispersion':MAX_DISP,'probability_rank_min':.80,'expected_excess_rank_min':.80}}
    results={}; qualified=[]
    for h in (1,5,10):
        bundle['models'][str(h)]={}; results[str(h)]={}
        for side in ('up','down'):
            label=f'label_{side}_{h}'
            required=[label,f'fwd_{h}_excess',f'fwd_{h}_return']
            tr=train.dropna(subset=required); pc=platt.dropna(subset=required); rc=rankcal.dropna(subset=required); ss=select.dropna(subset=required); dd=dev.dropna(subset=required)
            if min(len(tr),len(pc),len(rc),len(ss))<300 or tr[label].nunique()<2 or pc[label].nunique()<2:
                results[str(h)][side]={'status':'INSUFFICIENT_HISTORY'}; continue
            members=[]; p_rc=[]; p_ss=[]; p_dd=[]
            for name in ('logistic','hgb2','hgb4'):
                m=meta.factory(name); m.fit(tr[feats],tr[label].astype(int)); raw=np.clip(m.predict_proba(pc[feats])[:,1],1e-5,1-1e-5); cal=meta.platt(raw,pc[label].astype(int).to_numpy()); members.append({'name':name,'model':m,'calibrator':cal})
                p_rc.append(v8.calibrate(m,cal,rc[feats])); p_ss.append(v8.calibrate(m,cal,ss[feats])); p_dd.append(v8.calibrate(m,cal,dd[feats]) if len(dd) else np.array([]))
            def agg(arr):
                st=np.stack(arr); return st.mean(axis=0),st.std(axis=0)
            rc_p,rc_d=agg(p_rc); ss_p,ss_d=agg(p_ss); dd_p,dd_d=agg(p_dd) if len(dd) else (np.array([]),np.array([]))
            exreg=reg_factory(); exreg.fit(tr[feats],tr[f'fwd_{h}_excess'].astype(float)); rc_ex=exreg.predict(rc[feats]); ss_ex=exreg.predict(ss[feats]); dd_ex=exreg.predict(dd[feats]) if len(dd) else np.array([])
            retreg=reg_factory(); retreg.fit(tr[feats],tr[f'fwd_{h}_return'].astype(float))
            emp=empirical_rank_probability(rc,rc_p,rc_d,rc_ex,rc[label].astype(int).to_numpy(),side); ep=emp.get('shrunk_rate')
            sel,selpicks=pick_stats(ss,ss_p,ss_d,ss_ex,ss[label].astype(int).to_numpy(),side,ep)
            devs,devpicks=pick_stats(dd,dd_p,dd_d,dd_ex,dd[label].astype(int).to_numpy(),side,ep) if len(dd) else ({'signals':0,'precision':None,'base_rate':None},pd.DataFrame())
            q=v8.quality(ss[label].astype(int),ss_p)
            selection_pass=bool(sel['signals']>=MIN_SELECT_N and sel['precision'] is not None and sel['precision']>=.50 and sel['wilson_lower_95'] is not None and sel['wilson_lower_95']>sel['base_rate']+.03 and sel['calibration_gap'] is not None and sel['calibration_gap']<=.10)
            dev_pass=bool(devs['signals']>=MIN_DEV_N and devs['precision'] is not None and devs['precision']>devs['base_rate']+.03)
            rank_quality=bool((q['auc'] or 0)>=.53 or (q['brier']<q['base_brier'] and q['log_loss']<q['base_log_loss']))
            ok=bool(selection_pass and dev_pass and rank_quality)
            rstats=None
            if not selpicks.empty:
                zz=ss.loc[selpicks.index,[f'fwd_{h}_return',f'fwd_{h}_excess']].dropna()
                if len(zz)>=30:
                    rr=zz[f'fwd_{h}_return'].astype(float); ee=zz[f'fwd_{h}_excess'].astype(float); rstats={'n':int(len(zz)),'median_return':float(rr.median()),'q20_return':float(rr.quantile(.20)),'q80_return':float(rr.quantile(.80)),'median_excess':float(ee.median())}
            bundle['models'][str(h)][side]={'members':members,'excess_regressor':exreg,'return_regressor':retreg,'empirical_rank_probability':emp,'selection':sel,'development_diagnostic':devs,'return_stats':rstats,'qualified_for_forward':ok}
            results[str(h)][side]={'selection_probability_quality':q,'rank_calibration':emp,'selection_consensus':sel,'development_2026_diagnostic':devs,'return_stats':rstats,'selection_pass':selection_pass,'development_pass':dev_pass,'rank_quality_pass':rank_quality,'qualified_for_forward':ok}
            if ok:qualified.append({'horizon':h,'side':side})
    bundle['status']='FROZEN_FOR_PROSPECTIVE_VALIDATION' if qualified else 'STAGING_NO_RULE_QUALIFIED'
    joblib.dump(bundle,STAGE,compress=3)
    val={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'V9 QUALIFIED FOR FORWARD SHADOW' if qualified else 'V9 NO RANK-CONSENSUS RULE QUALIFIED','model_version':'rank-consensus-v9','dataset_rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'symbols':int(data.symbol.nunique()),'missing':missing,'validation_design':'pre-2025Q3 fit; Jul-Aug 2025 Platt; Sep 2025 rank-calibration; 2025Q4 frozen selection; 2026 diagnostic sanity only; final proof prospective','architecture':'side-specific 3-model calibrated ensemble + expected-excess regressor + probability-rank/excess-rank consensus + top-1 per snapshot','qualified':qualified,'results':results}; VAL.write_text(json.dumps(val,separators=(',',':')))
    STATUS.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'status':'READY_FOR_INFERENCE_WIRING' if qualified else 'NOT_PROMOTED','qualified':qualified,'stage_model_exists':STAGE.exists()},separators=(',',':')))
    print(json.dumps({'status':val['status'],'rows':len(data),'pool':len(pool),'qualified':qualified,'results':results},indent=2))

if __name__=='__main__':main()
