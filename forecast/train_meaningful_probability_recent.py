from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import forecast.train_meaningful_move_probability as mm


def recent_fit(data,h):
    target=f'y_{h}';d=data[data[target].notna()].reset_index(drop=True);X=d[mm.BASE_FEATURES].replace([np.inf,-np.inf],np.nan);y=d[target].astype(int);dates=pd.to_datetime(d.snapshot_date)
    calmask=(dates>=pd.Timestamp('2025-01-01'))&(dates<pd.Timestamp('2025-07-01'));select=(dates>=pd.Timestamp('2025-07-01'))&(dates<pd.Timestamp('2026-01-01'));final=dates>=pd.Timestamp('2026-01-01')
    Xc,yc=X[calmask],y[calmask];Xs,ys=X[select],y[select];Xf,yf_=X[final],y[final]
    candidates=[];trials=[]
    for window,start in [('full','2021-01-01'),('recent','2023-01-01')]:
        train=(dates>=pd.Timestamp(start))&(dates<pd.Timestamp('2025-01-01'));Xtr,ytr=X[train],y[train]
        for name,model in mm.models().items():
            model.fit(Xtr,ytr);cs=mm.calibrate(model,Xc,yc);ps=mm.apply(model,cs,Xs);tails={};score=-1
            for side,k in [('down',0),('up',2)]:
                best,_=mm.tail_choice(ys,ps,k);tails[side]=best
                if best:score=max(score,best['wilson_lower'])
            cl=float(log_loss(yc,mm.apply(model,cs,Xc),labels=[0,1,2]));trials.append({'window':window,'model':name,'tail_score':score,'cal_logloss':cl,'tails':tails});candidates.append((score,-cl,window,name,model,cs,tails,int(train.sum()),ytr))
    viable=[x for x in candidates if x[0]>=0];chosen=max(viable,key=lambda z:(z[0],z[1])) if viable else max(candidates,key=lambda z:z[1]);_,_,window,name,model,cs,tails,train_n,ytr=chosen;pf=mm.apply(model,cs,Xf)
    prior=np.bincount(ytr,minlength=3)/len(ytr);basep=np.tile(prior,(len(yf_),1));ll=float(log_loss(yf_,pf,labels=[0,1,2]));bll=float(log_loss(yf_,basep,labels=[0,1,2]));br=mm.mc_brier(yf_,pf);bbr=mm.mc_brier(yf_,basep);eces={'down':mm.class_ece(yf_,pf,0),'up':mm.class_ece(yf_,pf,2)};overall=ll<bll and br<bbr
    ff=d.loc[final,['datetime',target]].copy();ff['pdown']=pf[:,0];ff['pup']=pf[:,2];tests={};accepted=[]
    for side,k,col in [('down',0,'pdown'),('up',2,'pup')]:
        if not tails.get(side):tests[side]={'n':0,'accepted':False};continue
        x=mm.final_tail(yf_,pf,k,tails[side]['threshold']);x['precision1']=mm.precision_k(ff,col,target,k,1,tails[side]['threshold']);x['precision3']=mm.precision_k(ff,col,target,k,3,tails[side]['threshold']);x['accepted']=bool(overall and eces[side]<=.05 and x['accepted'] and x['precision3']['picks']>=60 and x['precision3']['wilson_lower']>x['base']+.02);tests[side]=x
        if x['accepted']:accepted.append(side)
    metrics={'horizon':h,'training_window':window,'model':name,'train_n':train_n,'cal_h1_2025_n':int(calmask.sum()),'selection_h2_2025_n':int(select.sum()),'final_2026_n':int(final.sum()),'base_class_rates':{'down':float(prior[0]),'neutral':float(prior[1]),'up':float(prior[2])},'multiclass_logloss':ll,'base_logloss':bll,'multiclass_brier':br,'base_brier':bbr,'ece':eces,'selection_thresholds':tails,'final_tests':tests,'accepted_sides':accepted,'accepted_for_display':bool(accepted),'model_trials':trials}
    return {'model':model,'calibrators':cs,'features':mm.BASE_FEATURES,'metrics':metrics,'thresholds':{s:(tails[s]['threshold'] if tails.get(s) else None) for s in ['up','down']}},metrics

mm.fit_h=recent_fit
if __name__=='__main__':mm.main()
