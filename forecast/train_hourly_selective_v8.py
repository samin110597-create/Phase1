from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import forecast.train_hourly_meta_model_v7 as base
from forecast.run_hourly_meta_v7 import fixed_asof_prev
import forecast.train_intraday_meta_model_v5 as meta

base.asof_prev = fixed_asof_prev

STAGE_MODEL=Path('forecast/data/hourly_selective_v8.joblib')
LIVE_MODEL=Path('forecast/data/intraday_signal_v3.joblib')
VALIDATION=Path('docs/data/hourly_selective_v8_validation.json')
STATUS=Path('docs/data/hourly_selective_v8_status.json')
for p in (STAGE_MODEL, VALIDATION, STATUS): p.parent.mkdir(parents=True, exist_ok=True)

MIN_SIGNALS_SELECT=60
MIN_SIGNALS_DEV=30
MIN_PRECISION=.55
MAX_CAL_GAP=.08
MAX_DISP=.10
MIN_ALIGN=4


def build_dataset():
    bars=base.download_hourly(base.ALL)
    missing=[s for s in base.ALL if bars.get(s,pd.DataFrame()).empty]
    if 'SPY' in missing: raise RuntimeError('SPY hourly history unavailable')
    daily={s:base.daily_from_hourly(bars.get(s,pd.DataFrame())) for s in base.ALL}
    market_ctx={}
    for bmk in base.MARKET:
        d=daily.get(bmk,pd.DataFrame())
        if not d.empty: market_ctx[bmk]=base.daily_features(d,bmk.replace('^','').lower())
    parts=[]; coverage={}
    spy_sn=base.snapshots('SPY',bars['SPY'])
    spy_cols=['h1_ret3','h1_ret6','h4_ret','session_ret','session_vwap_dist','session_rvol']
    spy_keep=spy_sn[['snapshot_dt']+spy_cols].rename(columns={c:'spy_'+c for c in spy_cols})
    for s in base.UNIVERSE:
        if bars.get(s,pd.DataFrame()).empty or daily.get(s,pd.DataFrame()).empty: continue
        x=base.snapshots(s,bars[s])
        x=base.asof_prev(x,base.daily_features(daily[s],'day'))
        x=base.asof_prev(x,base.weekly_features(daily[s]))
        sec=base.SECTOR_OF.get(s)
        if sec and not daily.get(sec,pd.DataFrame()).empty:
            sf=base.daily_features(daily[sec],'sector')
            x=base.asof_prev(x,sf[['sector_ret5','sector_ret20','sector_ret63','sector_above_ema20','sector_ema20_gt_ema50']])
        for bmk,f in market_ctx.items():
            cols=[c for c in f.columns if c.endswith(('ret5','ret20','ret63','atr_pct','rv20')) or 'above_ema20' in c or 'ema20_gt_ema50' in c]
            x=base.asof_prev(x,f[cols])
        x=x.merge(spy_keep,on='snapshot_dt',how='left')
        x['rel_h1_6_vs_spy']=x.h1_ret6-x.spy_h1_ret6
        x['rel_h4_vs_spy']=x.h4_ret-x.spy_h4_ret
        if 'sector_ret20' in x: x['rel20_vs_sector']=x.day_ret20-x.sector_ret20
        x=base.attach_labels(x,daily[s],daily['SPY'])
        coverage[s]={'rows':int(len(x)),'first':str(x.snapshot_dt.min()),'last':str(x.snapshot_dt.max())}
        parts.append(x)
    data=pd.concat(parts,ignore_index=True)
    data['candidate_score']=base.candidate_score(data)
    g=data.groupby('snapshot_dt')
    data['candidate_rank_pct']=g.candidate_score.rank(pct=True)
    data['activity_rank_pct']=g.session_ret.transform(lambda s:s.abs().rank(pct=True))
    data['rvol_rank_pct']=g.session_rvol.rank(pct=True)
    data['cross_section_rel_rank']=g.rel_h1_6_vs_spy.rank(pct=True)
    data=data.sort_values(['snapshot_dt','symbol']).reset_index(drop=True)
    return data, coverage, missing


def features(df):
    ignore={'snapshot_dt','slot','symbol','close'}
    pref=('fwd_','spy_fwd_','label_','move_threshold_')
    return [c for c in df.columns if c not in ignore and not c.startswith(pref) and pd.api.types.is_numeric_dtype(df[c])]


def structural_alignment(frame,side):
    up=(side=='up')
    cols=[]
    for c in ('h1_above_ema20','h1_ema20_gt_ema50','day_above_ema20','week_above_ema20'):
        v=frame[c].fillna(.5)>.5
        cols.append(v if up else ~v)
    h4=frame['h4_ret'].fillna(0)>=0
    cols.append(h4 if up else ~h4)
    arr=np.column_stack([x.to_numpy(dtype=bool) for x in cols])
    return arr.sum(axis=1)


def calibrate(model,calibrator,X):
    raw=np.clip(model.predict_proba(X)[:,1],1e-5,1-1e-5)
    z=np.log(raw/(1-raw)).reshape(-1,1)
    return calibrator.predict_proba(z)[:,1]


def quality(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); b=float(y.mean()); bp=np.full(len(y),b)
    try: auc=float(roc_auc_score(y,p))
    except Exception: auc=None
    return {'n':int(len(y)),'base_rate':b,'auc':auc,'brier':float(brier_score_loss(y,p)),'base_brier':float(brier_score_loss(y,bp)),'log_loss':float(log_loss(y,p,labels=[0,1])),'base_log_loss':float(log_loss(y,bp,labels=[0,1]))}


def wilson(s,n,z=1.96):
    if n<=0:return None
    p=s/n; den=1+z*z/n; center=p+z*z/(2*n); adj=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (center-adj)/den


def filtered(frame,p,disp,side,threshold):
    z=frame[['snapshot_dt','candidate_rank_pct']].copy(); z['p']=p; z['disp']=disp; z['align']=structural_alignment(frame,side)
    tail=z.candidate_rank_pct>=.75 if side=='up' else z.candidate_rank_pct<=.25
    z=z[tail & (z.p>=threshold) & (z.disp<=MAX_DISP) & (z['align']>=MIN_ALIGN)]
    if z.empty:return z
    return z.sort_values(['snapshot_dt','p'],ascending=[True,False]).groupby('snapshot_dt').head(1)


def choose_rule(frame,p,disp,y,side):
    y=np.asarray(y,int); base_rate=float(y.mean()); best=None
    for t in np.arange(.35,.76,.01):
        picks=filtered(frame,p,disp,side,float(t))
        if picks.empty: continue
        yy=pd.Series(y,index=frame.index); py=yy.loc[picks.index].astype(int)
        n=len(py); s=int(py.sum()); pr=s/n; lo=wilson(s,n); avg=float(picks.p.mean()); gap=abs(pr-avg)
        accepted=bool(n>=MIN_SIGNALS_SELECT and pr>=MIN_PRECISION and lo is not None and lo>base_rate+.03 and gap<=MAX_CAL_GAP)
        score=(20 if accepted else 0)+(lo or 0)*2+pr-gap*.5+min(n,250)/10000
        cand={'threshold':round(float(t),2),'signals':int(n),'precision':float(pr),'wilson_lower_95':float(lo),'mean_probability':avg,'calibration_gap':gap,'base_rate':base_rate,'accepted_selection':accepted,'score':score}
        if best is None or cand['score']>best['score']: best=cand
    if best: best.pop('score',None)
    return best


def test_rule(frame,p,disp,y,side,rule):
    if not rule:return {'signals':0,'precision':None,'base_rate':float(np.mean(y)),'sanity_pass':False}
    picks=filtered(frame,p,disp,side,float(rule['threshold']))
    if picks.empty:return {'signals':0,'precision':None,'base_rate':float(np.mean(y)),'sanity_pass':False}
    yy=pd.Series(np.asarray(y,int),index=frame.index); py=yy.loc[picks.index].astype(int)
    n=len(py); s=int(py.sum()); pr=s/n; base_rate=float(np.mean(y)); lo=wilson(s,n); avg=float(picks.p.mean())
    return {'signals':int(n),'precision':float(pr),'wilson_lower_95':float(lo),'mean_probability':avg,'calibration_gap':abs(pr-avg),'base_rate':base_rate,'sanity_pass':bool(n>=MIN_SIGNALS_DEV and pr>=max(.48,base_rate+.03))}


def return_stats(frame,p,disp,side,h,rule):
    if not rule:return None
    picks=filtered(frame,p,disp,side,float(rule['threshold']))
    if picks.empty:return None
    z=frame.loc[picks.index,[f'fwd_{h}_return',f'fwd_{h}_excess']].dropna()
    if len(z)<30:return None
    r=z[f'fwd_{h}_return'].astype(float); e=z[f'fwd_{h}_excess'].astype(float)
    return {'n':int(len(z)),'median_return':float(r.median()),'q20_return':float(r.quantile(.20)),'q80_return':float(r.quantile(.80)),'median_excess':float(e.median())}


def main():
    data,coverage,missing=build_dataset(); feats=features(data)
    pool=data[(data.candidate_rank_pct>=.75)|(data.candidate_rank_pct<=.25)].copy()
    periods={
        'train':pool.snapshot_dt<pd.Timestamp('2025-07-01'),
        'cal':(pool.snapshot_dt>=pd.Timestamp('2025-07-01'))&(pool.snapshot_dt<pd.Timestamp('2025-10-01')),
        'select':(pool.snapshot_dt>=pd.Timestamp('2025-10-01'))&(pool.snapshot_dt<pd.Timestamp('2026-01-01')),
        'dev2026':pool.snapshot_dt>=pd.Timestamp('2026-01-01')}
    bundle={'version':'hourly-selective-v8','trained_at':datetime.now(timezone.utc).isoformat(),'status':'STAGING_AWAITING_EVIDENCE_GATE','feature_columns':feats,'models':{},'decision_hours':base.DECISION_HOURS,'policy':{'min_alignment':MIN_ALIGN,'max_dispersion':MAX_DISP,'min_selection_precision':MIN_PRECISION}}
    results={}; any_qualified=False
    for h in (1,5,10):
        valid=pool[f'label_up_{h}'].notna()&pool[f'label_down_{h}'].notna(); ph=pool[valid].copy(); idx={k:ph.index[m.reindex(ph.index,fill_value=False)] for k,m in periods.items()}; split={k:ph.loc[v] for k,v in idx.items()}
        hr={'counts':{k:int(len(v)) for k,v in split.items()},'sides':{}}
        bundle['models'][str(h)]={}
        for side in ('up','down'):
            target=f'label_{side}_{h}'; ys={k:split[k][target].astype(int) for k in split}
            if min(len(split['train']),len(split['cal']),len(split['select']))<400 or ys['train'].nunique()<2 or ys['cal'].nunique()<2:
                hr['sides'][side]={'status':'INSUFFICIENT_HISTORY'}; continue
            members=[]; ps=[]; pdv=[]
            for name in ('logistic','hgb2','hgb4'):
                m=meta.factory(name); m.fit(split['train'][feats],ys['train']); raw=np.clip(m.predict_proba(split['cal'][feats])[:,1],1e-5,1-1e-5); cal=meta.platt(raw,ys['cal'].to_numpy())
                members.append({'name':name,'model':m,'calibrator':cal}); ps.append(calibrate(m,cal,split['select'][feats])); pdv.append(calibrate(m,cal,split['dev2026'][feats]) if len(split['dev2026']) else np.array([]))
            sel_stack=np.stack(ps); selp=sel_stack.mean(axis=0); seld=sel_stack.std(axis=0)
            if len(split['dev2026']): dev_stack=np.stack(pdv); devp=dev_stack.mean(axis=0); devd=dev_stack.std(axis=0)
            else: devp=np.array([]); devd=np.array([])
            qsel=quality(ys['select'],selp); rule=choose_rule(split['select'],selp,seld,ys['select'],side); dev=test_rule(split['dev2026'],devp,devd,ys['dev2026'],side,rule) if len(split['dev2026']) else {'signals':0,'sanity_pass':False}; stats=return_stats(split['select'],selp,seld,side,h,rule)
            quality_pass=bool(qsel['brier']<qsel['base_brier'] and qsel['log_loss']<qsel['base_log_loss'] and (qsel['auc'] or 0)>=.52)
            qualified=bool(rule and rule.get('accepted_selection') and dev.get('sanity_pass') and quality_pass)
            if qualified:any_qualified=True
            bundle['models'][str(h)][side]={'members':members,'threshold':float(rule['threshold']) if qualified else None,'selection_rule':rule,'development_diagnostic':dev,'return_stats':stats,'qualified_for_forward':qualified}
            hr['sides'][side]={'selection_probability_quality':qsel,'selection_rule':rule,'development_2026_diagnostic':dev,'historical_return_target':stats,'quality_pass':quality_pass,'qualified_for_forward':qualified}
        results[str(h)]=hr
    bundle['status']='FROZEN_FOR_PROSPECTIVE_VALIDATION' if any_qualified else 'STAGING_NO_RULE_QUALIFIED'
    joblib.dump(bundle,STAGE_MODEL,compress=3)
    if any_qualified: joblib.dump(bundle,LIVE_MODEL,compress=3)
    validation={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'V8 QUALIFIED FOR FORWARD SHADOW' if any_qualified else 'V8 NO SELECTIVE RULE QUALIFIED','model_version':'hourly-selective-v8','dataset_rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'symbols':int(data.symbol.nunique()),'missing':missing,'validation_design':'pre-2025Q3 fit; 2025Q3 Platt calibration; 2025Q4 frozen selection; 2026 diagnostic sanity only because previously inspected; final proof prospective','architecture':'side-specific calibrated 3-model ensemble + cross-sectional tail + 4/5 structural alignment + top-1-per-snapshot abstention','any_qualified_for_forward':any_qualified,'results':results}
    VALIDATION.write_text(json.dumps(validation,separators=(',',':')))
    STATUS.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'status':'PROMOTED_TO_LIVE_MODEL' if any_qualified else 'NOT_PROMOTED','stage_model_exists':STAGE_MODEL.exists(),'live_model_overwritten':any_qualified,'dataset_rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'qualified':[{ 'horizon':h,'side':s } for h,r in results.items() for s,z in r.get('sides',{}).items() if z.get('qualified_for_forward')]},separators=(',',':')))
    print(json.dumps({'status':validation['status'],'rows':len(data),'pool':len(pool),'qualified':json.loads(STATUS.read_text())['qualified'],'results':{h:{s:{'quality':z.get('selection_probability_quality'),'rule':z.get('selection_rule'),'dev':z.get('development_2026_diagnostic'),'qualified':z.get('qualified_for_forward')} for s,z in r.get('sides',{}).items()} for h,r in results.items()}},indent=2))

if __name__=='__main__':main()
