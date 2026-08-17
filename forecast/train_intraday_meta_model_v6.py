from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
import forecast.train_intraday_meta_model_v5 as b

DATA=b.DATA; MODEL=b.MODEL; VALIDATION=b.VALIDATION


def main():
    if not DATA.exists(): raise RuntimeError('intraday live-like dataset not built')
    df=pd.read_csv(DATA,compression='gzip',parse_dates=['snapshot_dt']); feats=b.features(df)
    pool=df[(df['candidate_rank_pct']>=.75)|(df['candidate_rank_pct']<=.25)].copy()
    periods={
        'train':pool.snapshot_dt<pd.Timestamp('2024-01-01'),
        'cal_platt':(pool.snapshot_dt>=pd.Timestamp('2024-01-01'))&(pool.snapshot_dt<pd.Timestamp('2024-07-01')),
        'meta_cal':(pool.snapshot_dt>=pd.Timestamp('2024-07-01'))&(pool.snapshot_dt<pd.Timestamp('2025-01-01')),
        'select':(pool.snapshot_dt>=pd.Timestamp('2025-01-01'))&(pool.snapshot_dt<pd.Timestamp('2026-01-01')),
        'dev2026':pool.snapshot_dt>=pd.Timestamp('2026-01-01'),
    }
    bundle={'version':'intraday-meta-v6','trained_at':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION','feature_columns':feats,'models':{},'meta_policy':{'min_local_n':b.MIN_LOCAL_N,'max_member_dispersion':b.MAX_DISPERSION,'prior_strength':b.PRIOR_STRENGTH},'chronology':'pre2024 fit; 2024H1 Platt; 2024H2 comparable-setup meta calibration; 2025 threshold selection; 2026 diagnostic only'}
    results={}
    for h in (1,5,10):
        valid=pool[f'label_up_{h}'].notna()&pool[f'label_down_{h}'].notna(); ph=pool[valid].copy(); y=b.make_target(ph,h)
        split={}; ys={}
        for name,mask in periods.items():
            ids=ph.index[mask.reindex(ph.index,fill_value=False)]; split[name]=ph.loc[ids]; ys[name]=y.loc[ids]
        if min(len(split['train']),len(split['cal_platt']),len(split['meta_cal']),len(split['select']))<350:
            results[str(h)]={'status':'INSUFFICIENT_HISTORY','counts':{k:int(len(v)) for k,v in split.items()}}; continue
        members=[]; pmeta_members=[]; psel_members=[]; pdev_members=[]
        for name in ('logistic','hgb2','hgb4'):
            m=b.factory(name); m.fit(split['train'][feats],ys['train'])
            raw=m.predict_proba(split['cal_platt'][feats]); cls=list(m.classes_); cals={}
            for k in b.CLASSES: cals[str(k)]=b.platt(raw[:,cls.index(k)],(ys['cal_platt'].to_numpy()==k).astype(int))
            members.append({'name':name,'model':m,'calibrators':cals})
            pmeta_members.append(b.calibrate_members(m,cals,split['meta_cal'][feats])); psel_members.append(b.calibrate_members(m,cals,split['select'][feats])); pdev_members.append(b.calibrate_members(m,cals,split['dev2026'][feats]) if len(split['dev2026']) else np.empty((0,3)))
        ens_meta=np.mean(np.stack(pmeta_members),axis=0); disp_meta=np.max(np.std(np.stack(pmeta_members),axis=0),axis=1)
        table,global_table=b.build_local_table(split['meta_cal'],ens_meta,ys['meta_cal'])
        ens_sel=np.mean(np.stack(psel_members),axis=0); disp_sel=np.max(np.std(np.stack(psel_members),axis=0),axis=1); meta_sel,nsel=b.meta_adjust(split['select'],ens_sel,table,global_table)
        if len(split['dev2026']):
            ens_dev=np.mean(np.stack(pdev_members),axis=0); disp_dev=np.max(np.std(np.stack(pdev_members),axis=0),axis=1); meta_dev,ndev=b.meta_adjust(split['dev2026'],ens_dev,table,global_table)
        else: meta_dev=np.empty((0,3)); ndev=np.empty((0,3),int); disp_dev=np.array([])
        # For target magnitudes, use the independent 2024H2 block and the threshold frozen from 2025.
        meta_meta,nmeta=b.meta_adjust(split['meta_cal'],ens_meta,table,global_table)
        rules={}; devtests={}; rstats={}
        for side in ('up','down'):
            rules[side]=b.side_threshold(split['select'],meta_sel,nsel,disp_sel,ys['select'],side)
            devtests[side]=b.test_threshold(split['dev2026'],meta_dev,ndev,disp_dev,ys['dev2026'],side,rules[side]) if len(split['dev2026']) else {'signals':0}
            rstats[side]=b.return_stats(split['meta_cal'],meta_meta,nmeta,disp_meta,ys['meta_cal'],side,h,rules[side])
        bundle['models'][str(h)]={'members':members,'local_table':table,'global_table':global_table,'thresholds':{s:(rules[s]['threshold'] if rules[s] else None) for s in ('up','down')},'selection_rules':rules,'return_stats':rstats}
        results[str(h)]={'status':'FROZEN_META_MODEL','train_n':int(len(split['train'])),'platt_calibration_n':int(len(split['cal_platt'])),'meta_calibration_n':int(len(split['meta_cal'])),'selection_2025_n':int(len(split['select'])),'development_2026_n':int(len(split['dev2026'])),'selection_probability_quality':b.multiclass_metrics(ys['select'],meta_sel),'development_2026_probability_quality':b.multiclass_metrics(ys['dev2026'],meta_dev) if len(split['dev2026']) else {'n':0},'selection_signal_rules':rules,'development_2026_diagnostic':devtests,'historical_return_targets':rstats,'feature_count':len(feats),'cross_section_features_used':[c for c in ('candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank') if c in feats],'truth_note':'2025 thresholds are evaluated against a comparable-setup table learned only from 2024H2. 2026 remains diagnostic because it was previously inspected. Final validation is prospective.'}
    joblib.dump(bundle,MODEL,compress=3)
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'INTRADAY META MODEL V6 FROZEN — LEAKAGE-CONTROLLED, SELECTIVE, REGIME-AWARE, AWAITING PROSPECTIVE VALIDATION','dataset_rows':int(len(df)),'candidate_pool_rows':int(len(pool)),'symbols':int(df.symbol.nunique()),'features':feats,'model_file':str(MODEL),'architecture':'3-class 3-model ensemble + chronological Platt calibration + later-period regime/probability empirical shrinkage + cross-sectional ranks + abstention','validation_design':'pre-2024 model fit; 2024H1 Platt calibration; 2024H2 comparable-setup calibration; 2025 frozen threshold selection; 2026 diagnostic only; final proof prospective','signal_policy':'Signal requires frozen threshold, >=60 comparable historical observations, <=9% model-member dispersion, top/bottom-quartile cross-sectional rank, then live MTF alignment.','results':results}
    VALIDATION.write_text(json.dumps(summary,separators=(',',':')))
    print(json.dumps({'status':summary['status'],'rows':summary['dataset_rows'],'pool_rows':summary['candidate_pool_rows'],'symbols':summary['symbols'],'results':{h:{'up':v.get('selection_signal_rules',{}).get('up'),'down':v.get('selection_signal_rules',{}).get('down'),'dev':v.get('development_2026_diagnostic')} for h,v in results.items()}},indent=2))

if __name__=='__main__': main()
