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

DATA=Path('forecast/data/intraday_live_like_v2.csv.gz')
MODEL=Path('forecast/data/intraday_signal_v3.joblib')
VALIDATION=Path('docs/data/intraday_model_v3_validation.json')
MODEL.parent.mkdir(parents=True,exist_ok=True); VALIDATION.parent.mkdir(parents=True,exist_ok=True)

ID={'snapshot_dt','slot','symbol','close','split_like_gap','decision_price_proxy'}
DROP_PREFIX=('fwd_','spy_fwd_','label_','move_threshold_')
CLASSES=np.array([0,1,2]) # DOWN, NEUTRAL, UP
CLASS_NAME={0:'down',1:'neutral',2:'up'}
PRIOR_STRENGTH=40.0
MIN_LOCAL_N=60
MAX_DISPERSION=0.09


def features(df):
    out=[]
    for c in df.columns:
        if c in ID or c.startswith(DROP_PREFIX): continue
        if pd.api.types.is_numeric_dtype(df[c]): out.append(c)
    return out


def factory(name):
    if name=='logistic':
        return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=.30,max_iter=900,class_weight='balanced',solver='lbfgs'))])
    if name=='hgb2':
        return Pipeline([('impute',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(max_iter=240,learning_rate=.04,max_leaf_nodes=15,l2_regularization=3.0,min_samples_leaf=100,random_state=11))])
    if name=='hgb4':
        return Pipeline([('impute',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(max_iter=300,learning_rate=.03,max_leaf_nodes=31,l2_regularization=6.0,min_samples_leaf=140,random_state=19))])
    raise ValueError(name)


def make_target(df,h):
    up=df[f'label_up_{h}']; dn=df[f'label_down_{h}']
    y=np.where(up==1,2,np.where(dn==1,0,1)).astype(int)
    return pd.Series(y,index=df.index)


def platt(raw,y):
    p=np.clip(np.asarray(raw,float),1e-5,1-1e-5); x=np.log(p/(1-p)).reshape(-1,1)
    m=LogisticRegression(C=1.0,max_iter=500); m.fit(x,np.asarray(y,int)); return m


def calibrate_members(model,calibrators,X):
    raw=model.predict_proba(X)
    cls=list(model.classes_)
    out=np.zeros((len(X),3),float)
    for k in CLASSES:
        j=cls.index(k); p=np.clip(raw[:,j],1e-5,1-1e-5); x=np.log(p/(1-p)).reshape(-1,1)
        out[:,k]=calibrators[str(k)].predict_proba(x)[:,1]
    s=out.sum(axis=1,keepdims=True); s[s<=0]=1
    return out/s


def regime_key(row):
    def bit(name):
        v=row.get(name,np.nan); return '1' if pd.notna(v) and float(v)>.5 else '0'
    return '|'.join([bit('spy_above_ema20'),bit('vix_above_ema20'),bit('sector_above_ema20'),str(row.get('slot','NA'))])


def pbin(p): return int(max(0,min(9,math.floor(float(p)*10))))


def build_local_table(frame,probs,y):
    table={}; global_table={}
    for i,(idx,row) in enumerate(frame.iterrows()):
        reg=regime_key(row)
        for k in CLASSES:
            b=pbin(probs[i,k]); yy=1 if int(y.loc[idx])==k else 0
            for key,store in ((f'{reg}|{k}|{b}',table),(f'{k}|{b}',global_table)):
                z=store.setdefault(key,{'n':0,'success':0}); z['n']+=1; z['success']+=yy
    for store in (table,global_table):
        for z in store.values(): z['rate']=z['success']/z['n'] if z['n'] else None
    return table,global_table


def meta_adjust(frame,probs,table,global_table):
    out=np.zeros_like(probs); local_n=np.zeros((len(frame),3),int)
    for i,(_,row) in enumerate(frame.iterrows()):
        reg=regime_key(row)
        for k in CLASSES:
            b=pbin(probs[i,k]); z=table.get(f'{reg}|{k}|{b}')
            if not z or z['n']<MIN_LOCAL_N: z=global_table.get(f'{k}|{b}')
            n=int(z['n']) if z else 0; rate=float(z['rate']) if z and z.get('rate') is not None else float(probs[i,k])
            out[i,k]=(n*rate+PRIOR_STRENGTH*float(probs[i,k]))/(n+PRIOR_STRENGTH) if n else float(probs[i,k]); local_n[i,k]=n
        s=out[i].sum(); out[i]=out[i]/s if s>0 else np.array([1/3,1/3,1/3])
    return out,local_n


def multiclass_metrics(y,p):
    Y=np.eye(3)[np.asarray(y,int)]
    base=np.bincount(np.asarray(y,int),minlength=3)/len(y); bp=np.tile(base,(len(y),1))
    return {'n':int(len(y)),'brier':float(np.mean(np.sum((p-Y)**2,axis=1))), 'base_brier':float(np.mean(np.sum((bp-Y)**2,axis=1))), 'log_loss':float(log_loss(y,p,labels=[0,1,2])), 'base_log_loss':float(log_loss(y,bp,labels=[0,1,2])), 'class_rates':{'down':float(base[0]),'neutral':float(base[1]),'up':float(base[2])}}


def wilson(s,n,z=1.96):
    if n<=0:return None
    p=s/n; den=1+z*z/n; center=p+z*z/(2*n); adj=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (center-adj)/den


def side_threshold(frame,p,local_n,disp,y,side):
    k=2 if side=='up' else 0; base=float(np.mean(np.asarray(y)==k)); best=None
    for t in np.arange(.42,.76,.01):
        z=frame[['snapshot_dt','candidate_rank_pct']].copy(); z['p']=p[:,k]; z['nloc']=local_n[:,k]; z['disp']=disp
        z['y']=(np.asarray(y)==k).astype(int)
        side_ok=z['candidate_rank_pct']>=.75 if side=='up' else z['candidate_rank_pct']<=.25
        z=z[side_ok & (z['p']>=t) & (z['nloc']>=MIN_LOCAL_N) & (z['disp']<=MAX_DISPERSION)]
        if z.empty: continue
        picks=z.sort_values(['snapshot_dt','p'],ascending=[True,False]).groupby('snapshot_dt').head(1)
        n=len(picks)
        if n<60: continue
        s=int(picks['y'].sum()); pr=s/n; lo=wilson(s,n); avg=float(picks['p'].mean()); gap=abs(pr-avg)
        accepted=bool(n>=80 and pr>=.60 and lo is not None and lo>base+.02 and gap<=.06)
        score=(1 if accepted else 0)*10 + (lo or 0)*2 + pr - gap*.5 + min(n,300)/10000
        cand={'threshold':round(float(t),2),'signals':int(n),'precision':float(pr),'wilson_lower_95':float(lo),'mean_probability':avg,'calibration_gap':gap,'base_rate':base,'accepted_2025':accepted,'score':score}
        if best is None or cand['score']>best['score']: best=cand
    if best: best.pop('score',None)
    return best


def test_threshold(frame,p,local_n,disp,y,side,rule):
    if not rule:return {'signals':0,'precision':None,'accepted_diagnostic':False}
    k=2 if side=='up' else 0; z=frame[['snapshot_dt','candidate_rank_pct']].copy(); z['p']=p[:,k]; z['nloc']=local_n[:,k]; z['disp']=disp; z['y']=(np.asarray(y)==k).astype(int)
    side_ok=z['candidate_rank_pct']>=.75 if side=='up' else z['candidate_rank_pct']<=.25
    z=z[side_ok & (z['p']>=float(rule['threshold'])) & (z['nloc']>=MIN_LOCAL_N) & (z['disp']<=MAX_DISPERSION)]
    picks=z.sort_values(['snapshot_dt','p'],ascending=[True,False]).groupby('snapshot_dt').head(1)
    if picks.empty:return {'signals':0,'precision':None,'accepted_diagnostic':False}
    n=len(picks); s=int(picks['y'].sum()); pr=s/n; base=float(np.mean(np.asarray(y)==k)); lo=wilson(s,n); avg=float(picks['p'].mean())
    return {'signals':int(n),'precision':float(pr),'wilson_lower_95':float(lo),'mean_probability':avg,'calibration_gap':abs(pr-avg),'base_rate':base,'accepted_diagnostic':bool(n>=40 and pr>base+.01)}


def return_stats(frame,p,local_n,disp,y,side,h,rule):
    k=2 if side=='up' else 0; ret=f'fwd_{h}_return'; ex=f'fwd_{h}_excess'
    if ret not in frame or ex not in frame:return None
    z=frame[['snapshot_dt','candidate_rank_pct',ret,ex]].copy(); z['p']=p[:,k]; z['nloc']=local_n[:,k]; z['disp']=disp; z['y']=(np.asarray(y)==k).astype(int)
    side_ok=z['candidate_rank_pct']>=.75 if side=='up' else z['candidate_rank_pct']<=.25
    th=float(rule['threshold']) if rule else .50
    z=z[side_ok & (z['p']>=th) & (z['nloc']>=MIN_LOCAL_N) & (z['disp']<=MAX_DISPERSION)].dropna(subset=[ret,ex])
    if len(z)<30:return None
    vals=z[ret].astype(float); exc=z[ex].astype(float)
    return {'n':int(len(z)),'median_return':float(vals.median()),'q20_return':float(vals.quantile(.20)),'q80_return':float(vals.quantile(.80)),'median_excess':float(exc.median())}


def main():
    if not DATA.exists(): raise RuntimeError('intraday live-like dataset not built')
    df=pd.read_csv(DATA,compression='gzip',parse_dates=['snapshot_dt']); feats=features(df)
    # Historical candidate pool mirrors the top/bottom quartile shortlist idea.
    pool=df[(df['candidate_rank_pct']>=.75)|(df['candidate_rank_pct']<=.25)].copy()
    periods={'train':pool.snapshot_dt<pd.Timestamp('2024-01-01'),'cal':(pool.snapshot_dt>=pd.Timestamp('2024-01-01'))&(pool.snapshot_dt<pd.Timestamp('2025-01-01')),'select':(pool.snapshot_dt>=pd.Timestamp('2025-01-01'))&(pool.snapshot_dt<pd.Timestamp('2026-01-01')),'dev2026':pool.snapshot_dt>=pd.Timestamp('2026-01-01')}
    bundle={'version':'intraday-meta-v5','trained_at':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION','feature_columns':feats,'models':{},'meta_policy':{'min_local_n':MIN_LOCAL_N,'max_member_dispersion':MAX_DISPERSION,'prior_strength':PRIOR_STRENGTH}}
    results={}
    for h in (1,5,10):
        valid=pool[f'label_up_{h}'].notna() & pool[f'label_down_{h}'].notna(); ph=pool[valid].copy(); y=make_target(ph,h)
        idx={k:ph.index[periods[k].reindex(ph.index,fill_value=False)] for k in periods}; split={k:ph.loc[v] for k,v in idx.items()}; ys={k:y.loc[v] for k,v in idx.items()}
        if min(len(split['train']),len(split['cal']),len(split['select']))<500:
            results[str(h)]={'status':'INSUFFICIENT_HISTORY'}; continue
        members=[]; psel_members=[]; pdev_members=[]
        for name in ('logistic','hgb2','hgb4'):
            m=factory(name); m.fit(split['train'][feats],ys['train'])
            raw=m.predict_proba(split['cal'][feats]); cls=list(m.classes_); cals={}
            for k in CLASSES: cals[str(k)]=platt(raw[:,cls.index(k)],(ys['cal'].to_numpy()==k).astype(int))
            psel=calibrate_members(m,cals,split['select'][feats]); pdev=calibrate_members(m,cals,split['dev2026'][feats]) if len(split['dev2026']) else np.empty((0,3))
            members.append({'name':name,'model':m,'calibrators':cals}); psel_members.append(psel); pdev_members.append(pdev)
        ens_sel=np.mean(np.stack(psel_members),axis=0); disp_sel=np.max(np.std(np.stack(psel_members),axis=0),axis=1)
        table,global_table=build_local_table(split['select'],ens_sel,ys['select']); meta_sel,nsel=meta_adjust(split['select'],ens_sel,table,global_table)
        if len(split['dev2026']):
            ens_dev=np.mean(np.stack(pdev_members),axis=0); disp_dev=np.max(np.std(np.stack(pdev_members),axis=0),axis=1); meta_dev,ndev=meta_adjust(split['dev2026'],ens_dev,table,global_table)
        else: ens_dev=np.empty((0,3)); disp_dev=np.array([]); meta_dev=np.empty((0,3)); ndev=np.empty((0,3),int)
        rules={}; devtests={}; rstats={}
        for side in ('up','down'):
            rules[side]=side_threshold(split['select'],meta_sel,nsel,disp_sel,ys['select'],side)
            devtests[side]=test_threshold(split['dev2026'],meta_dev,ndev,disp_dev,ys['dev2026'],side,rules[side]) if len(split['dev2026']) else {'signals':0}
            rstats[side]=return_stats(split['select'],meta_sel,nsel,disp_sel,ys['select'],side,h,rules[side])
        bundle['models'][str(h)]={'members':members,'local_table':table,'global_table':global_table,'thresholds':{s:(rules[s]['threshold'] if rules[s] else None) for s in ('up','down')},'selection_rules':rules,'return_stats':rstats}
        results[str(h)]={'status':'FROZEN_META_MODEL','train_n':len(split['train']),'calibration_n':len(split['cal']),'selection_2025_n':len(split['select']),'development_2026_n':len(split['dev2026']),'selection_probability_quality':multiclass_metrics(ys['select'],meta_sel),'development_2026_probability_quality':multiclass_metrics(ys['dev2026'],meta_dev) if len(split['dev2026']) else {'n':0},'selection_signal_rules':rules,'development_2026_diagnostic':devtests,'historical_return_targets':rstats,'feature_count':len(feats),'cross_section_features_used':[c for c in ('candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank') if c in feats],'truth_note':'2026 is diagnostic because it was previously inspected. Final validation remains prospective after this frozen model is deployed.'}
    joblib.dump(bundle,MODEL,compress=3)
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'INTRADAY META MODEL V5 FROZEN — SELECTIVE, REGIME-AWARE, AWAITING PROSPECTIVE VALIDATION','dataset_rows':int(len(df)),'candidate_pool_rows':int(len(pool)),'symbols':int(df.symbol.nunique()),'features':feats,'model_file':str(MODEL),'architecture':'3-model multiclass ensemble + chronological Platt calibration + regime/probability empirical shrinkage + cross-sectional candidate ranks + abstention','validation_design':'pre-2024 train; 2024 calibration; 2025 frozen threshold/meta lookup construction; 2026 diagnostic only; final proof prospective','signal_policy':'A signal requires frozen side threshold, >=60 comparable historical bucket observations, <=9% model-member dispersion, and top/bottom-quartile cross-sectional rank. Otherwise abstain.','results':results}
    VALIDATION.write_text(json.dumps(summary,separators=(',',':')))
    print(json.dumps({'status':summary['status'],'rows':summary['dataset_rows'],'pool_rows':summary['candidate_pool_rows'],'symbols':summary['symbols'],'results':{h:{'up':v.get('selection_signal_rules',{}).get('up'),'down':v.get('selection_signal_rules',{}).get('down'),'dev':v.get('development_2026_diagnostic')} for h,v in results.items()}},indent=2))

if __name__=='__main__': main()
