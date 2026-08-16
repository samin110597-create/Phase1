from __future__ import annotations

import json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import forecast.train_live_probability as base

MODEL_OUT=Path('forecast/data/meaningful_move_probability.joblib')
VAL_OUT=Path('docs/data/meaningful_probability_validation.json')
SECTOR={'AAPL':'XLK','MSFT':'XLK','NVDA':'SMH','AMD':'SMH','AVGO':'SMH','MU':'SMH','AMZN':'XLY','TSLA':'XLY','META':'XLC','GOOGL':'XLC','JPM':'XLF','BAC':'XLF'}
CONTEXT=['SPY','QQQ','XLK','SMH','XLY','XLC','XLF','^VIX']
HORIZONS=[1,5,10]
THRESHOLDS=[.50,.55,.60,.65,.70,.75]


def raw_daily(symbols):
    raw=yf.download(symbols,start='2019-01-01',auto_adjust=False,actions=False,group_by='ticker',threads=True,progress=False)
    out={}
    for s in symbols:
        try:
            d=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy(); d=d.rename(columns=str.lower).dropna(subset=['close'])
            d.index=pd.to_datetime(d.index).tz_localize(None).astype('datetime64[ns]'); out[s]=d[['open','high','low','close','volume']].copy()
        except Exception: out[s]=pd.DataFrame()
    return out


def daily_feats(d,prefix):
    x=d.reset_index().rename(columns={d.index.name or 'index':'datetime'}); x['datetime']=pd.to_datetime(x['datetime']).astype('datetime64[ns]')
    f=base.add_features(x,prefix); f.index=x['datetime'].dt.normalize(); cols=[c for c in f.columns if c.startswith(prefix+'_')]
    return f[cols].shift(1)


def session_features(x):
    z=x.copy(); z['date']=z.datetime.dt.normalize(); g=z.groupby('date',sort=False)
    z['session_open']=g['open'].transform('first'); z['session_high']=g['high'].cummax(); z['session_low']=g['low'].cummin()
    z['session_from_open']=z['close']/z['session_open']-1
    rng=z['session_high']-z['session_low']; z['range_position']=(z['close']-z['session_low'])/rng.replace(0,np.nan)
    z['range_sofar_pct']=rng/z['close']
    return z


def stock_rows(symbol,intra,daily,spy_i,ctx_daily):
    if intra.empty or daily.empty:return pd.DataFrame()
    z=base.add_features(session_features(intra),'m15')
    c=z.close;v=z.volume.fillna(0);z['h4_ret']=c.pct_change(16);z['h4_ret3']=c.pct_change(48);z['h4_vol_ratio']=v.rolling(16).mean()/v.rolling(80).mean().replace(0,np.nan);z['h4_range_pct']=(z.high.rolling(16).max()-z.low.rolling(16).min())/c
    z['snapshot_date']=z.datetime.dt.normalize().astype('datetime64[ns]');z['minute']=z.datetime.dt.hour*60+z.datetime.dt.minute;z=z[z.minute.isin(base.ANCHOR_MINUTES)].copy()
    d=daily.copy();d.index=pd.to_datetime(d.index).astype('datetime64[ns]').normalize();z=z.join(daily_feats(d,'day'),on='snapshot_date')
    prev=d.close.shift(1);z['prev_close']=z.snapshot_date.map(prev);z['gap_pct']=z.session_open/z.prev_close-1;z['session_change']=z.close/z.prev_close-1;z['minute_norm']=(z.minute-570)/390
    sf=spy_i[['datetime','spy_ret3','spy_ret10','spy_rsi14','spy_above20','spy_macd','spy_atr','spy_vwap','spy_session','spy_range_pos']];z=z.merge(sf,on='datetime',how='left');z['rel_m15_vs_spy']=z.m15_ret10-z.spy_ret10
    for name,tab in ctx_daily.items():z=z.join(tab,on='snapshot_date')
    sec=SECTOR[symbol].lower(); z['rel_day3_vs_sector']=z.day_ret3-z[f'{sec}_ret3'];z['rel_day10_vs_sector']=z.day_ret10-z[f'{sec}_ret10']
    for h in HORIZONS:
        future=d.close.shift(-h);z[f'fwd_{h}']=z.snapshot_date.map(future)/z.close-1
        atr=z.day_atr_pct.clip(lower=.005,upper=.15);thr=(.35*atr*np.sqrt(h)).clip(lower=.006*np.sqrt(h),upper=.08);z[f'move_thr_{h}']=thr
        y=np.where(z[f'fwd_{h}']>thr,2,np.where(z[f'fwd_{h}']<-thr,0,1)).astype(float);y[pd.isna(z[f'fwd_{h}'])]=np.nan;z[f'y_{h}']=y
    z['symbol']=symbol;return z


def spy_intraday(df):
    z=base.add_features(session_features(df),'spyx');return pd.DataFrame({'datetime':z.datetime,'spy_ret3':z.spyx_ret3,'spy_ret10':z.spyx_ret10,'spy_rsi14':z.spyx_rsi14,'spy_above20':z.spyx_above_ema20,'spy_macd':z.spyx_macd_delta_pct,'spy_atr':z.spyx_atr_pct,'spy_vwap':z.spyx_vwap20_dist,'spy_session':z.session_from_open,'spy_range_pos':z.range_position})


def context_tables(daily):
    out={}
    for s,p in [('SPY','spyd'),('QQQ','qqqd'),('XLK','xlk'),('SMH','smh'),('XLY','xly'),('XLC','xlc'),('XLF','xlf')]:
        f=daily_feats(daily[s],p);keep=[f'{p}_ret3',f'{p}_ret10',f'{p}_rsi14',f'{p}_above_ema20',f'{p}_atr_pct'];out[s]=f[keep]
    v=daily['^VIX'].copy();v.index=pd.to_datetime(v.index).normalize();vf=pd.DataFrame(index=v.index);vf['vix_close']=v.close.shift(1);vf['vix_ret5']=v.close.pct_change(5).shift(1);out['VIX']=vf
    return out


def add_cross_section(df):
    g=df.groupby('datetime')
    df['rank_rel_spy']=g.rel_m15_vs_spy.rank(pct=True);df['rank_session']=g.session_change.rank(pct=True);df['rank_volume']=g.m15_vol_ratio20.rank(pct=True)
    df['breadth_m15']=g.m15_above_ema20.transform('mean');df['avg_session_change']=g.session_change.transform('mean');df['dispersion_session']=g.session_change.transform('std')
    return df

BASE_FEATURES=['m15_ret3','m15_ret10','m15_rsi14','m15_above_ema20','m15_ema20_gt_ema50','m15_macd_delta_pct','m15_atr_pct','m15_vol_ratio20','m15_vwap20_dist','h4_ret','h4_ret3','h4_vol_ratio','h4_range_pct','day_ret3','day_ret10','day_rsi14','day_above_ema20','day_ema20_gt_ema50','day_macd_delta_pct','day_atr_pct','day_vol_ratio20','day_vwap20_dist','gap_pct','session_change','session_from_open','range_position','range_sofar_pct','minute_norm','spy_ret3','spy_ret10','spy_rsi14','spy_above20','spy_macd','spy_atr','spy_vwap','spy_session','spy_range_pos','rel_m15_vs_spy','spyd_ret3','spyd_ret10','spyd_rsi14','spyd_above_ema20','spyd_atr_pct','qqqd_ret3','qqqd_ret10','qqqd_rsi14','qqqd_above_ema20','qqqd_atr_pct','vix_close','vix_ret5','rel_day3_vs_sector','rel_day10_vs_sector','rank_rel_spy','rank_session','rank_volume','breadth_m15','avg_session_change','dispersion_session']
for p in ['xlk','smh','xly','xlc','xlf']:BASE_FEATURES += [f'{p}_ret3',f'{p}_ret10',f'{p}_rsi14',f'{p}_above_ema20',f'{p}_atr_pct']


def models():return {'logistic':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',LogisticRegression(C=.4,max_iter=1500,random_state=42))]),'hgb2':Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingClassifier(max_depth=2,learning_rate=.04,max_iter=240,l2_regularization=2,random_state=42))]),'hgb4':Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingClassifier(max_depth=4,learning_rate=.04,max_iter=240,l2_regularization=2,random_state=42))]),'extra':Pipeline([('imp',SimpleImputer(strategy='median')),('m',ExtraTreesClassifier(n_estimators=400,max_depth=10,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=42))]),'rf':Pipeline([('imp',SimpleImputer(strategy='median')),('m',RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=42))])}


def calibrate(model,Xc,yc):
    raw=np.clip(model.predict_proba(Xc),1e-5,1-1e-5); cs=[]
    for k in range(3):
        lr=LogisticRegression(C=1e6,max_iter=500,random_state=42);lr.fit(base.safe_logit(raw[:,k]),(yc==k).astype(int));cs.append(lr)
    return cs

def apply(model,cs,X):
    raw=np.clip(model.predict_proba(X),1e-5,1-1e-5);q=np.column_stack([cs[k].predict_proba(base.safe_logit(raw[:,k]))[:,1] for k in range(3)]);return q/q.sum(axis=1,keepdims=True)

def mc_brier(y,p):
    oh=np.eye(3)[np.asarray(y,dtype=int)];return float(np.mean(np.sum((p-oh)**2,axis=1)))

def class_ece(y,p,k):return base.ece_and_bins((np.asarray(y)==k).astype(int),p[:,k])[0]

def tail_choice(y,p,k):
    base_rate=float(np.mean(np.asarray(y)==k));best=None;allx=[]
    for t in THRESHOLDS:
        m=p[:,k]>=t;n=int(m.sum())
        if n<150:continue
        obs=float(np.mean(np.asarray(y)[m]==k));pred=float(p[m,k].mean());low=base.wilson_lower(obs,n);gap=abs(obs-pred);x={'threshold':t,'n':n,'observed':obs,'mean_pred':pred,'wilson_lower':low,'gap':gap,'base':base_rate};allx.append(x)
        if obs>=.50 and low>base_rate+.03 and gap<=.06 and (best is None or (low,n)>(best['wilson_lower'],best['n'])):best=x
    return best,allx

def final_tail(y,p,k,t):
    br=float(np.mean(np.asarray(y)==k));m=p[:,k]>=t;n=int(m.sum())
    if not n:return {'n':0,'base':br,'accepted':False}
    obs=float(np.mean(np.asarray(y)[m]==k));pred=float(p[m,k].mean());low=base.wilson_lower(obs,n);gap=abs(obs-pred);return {'n':n,'base':br,'observed':obs,'mean_pred':pred,'wilson_lower':low,'gap':gap,'accepted':bool(n>=60 and obs>=.50 and low>br+.02 and gap<=.06)}

def precision_k(frame,probcol,label,kclass,k,t):
    vals=[]
    for _,g in frame.groupby('datetime'):
        x=g[g[probcol]>=t].nlargest(k,probcol)
        vals += (x[label].astype(int)==kclass).astype(float).tolist()
    if not vals:return {'picks':0,'precision':None,'wilson_lower':None}
    o=float(np.mean(vals));return {'picks':len(vals),'precision':o,'wilson_lower':base.wilson_lower(o,len(vals))}


def fit_h(data,h):
    target=f'y_{h}';d=data[data[target].notna()].reset_index(drop=True);X=d[BASE_FEATURES].replace([np.inf,-np.inf],np.nan);y=d[target].astype(int);dates=pd.to_datetime(d.snapshot_date)
    tr=dates<pd.Timestamp('2024-01-01');ca=(dates>=pd.Timestamp('2024-01-01'))&(dates<pd.Timestamp('2025-01-01'));se=(dates>=pd.Timestamp('2025-01-01'))&(dates<pd.Timestamp('2026-01-01'));fi=dates>=pd.Timestamp('2026-01-01')
    Xtr,ytr=X[tr],y[tr];Xc,yc=X[ca],y[ca];Xs,ys=X[se],y[se];Xf,yf_=X[fi],y[fi]
    base_prior=np.bincount(ytr,minlength=3)/len(ytr);base_p=np.tile(base_prior,(len(yf_),1));trials=[];fitted=[]
    for n,m in models().items():
        m.fit(Xtr,ytr);cs=calibrate(m,Xc,yc);ps=apply(m,cs,Xs);tails={};score=-1
        for side,k in [('down',0),('up',2)]:
            best,allx=tail_choice(ys,ps,k);tails[side]=best
            if best:score=max(score,best['wilson_lower'])
        trials.append({'model':n,'tail_score':score,'tails':tails});fitted.append((score,n,m,cs,tails))
    viable=[x for x in fitted if x[0]>=0];chosen=max(viable,key=lambda z:z[0]) if viable else fitted[0];_,name,m,cs,tails=chosen;pf=apply(m,cs,Xf)
    ll=float(log_loss(yf_,pf,labels=[0,1,2]));bll=float(log_loss(yf_,base_p,labels=[0,1,2]));br=mc_brier(yf_,pf);bbr=mc_brier(yf_,base_p);eces={'down':class_ece(yf_,pf,0),'up':class_ece(yf_,pf,2)};overall=ll<bll and br<bbr
    ff=d.loc[fi,['datetime',target]].copy();ff['pdown']=pf[:,0];ff['pup']=pf[:,2];final={};accepted=[]
    for side,k,col in [('down',0,'pdown'),('up',2,'pup')]:
        if not tails.get(side):final[side]={'n':0,'accepted':False};continue
        x=final_tail(yf_,pf,k,tails[side]['threshold']);x['precision1']=precision_k(ff,col,target,k,1,tails[side]['threshold']);x['precision3']=precision_k(ff,col,target,k,3,tails[side]['threshold']);x['accepted']=bool(overall and eces[side]<=.05 and x['accepted'] and x['precision3']['picks']>=60 and x['precision3']['wilson_lower']>x['base']+.02)
        final[side]=x
        if x['accepted']:accepted.append(side)
    metrics={'horizon':h,'model':name,'train_n':int(tr.sum()),'cal_n':int(ca.sum()),'select_n':int(se.sum()),'final_n':int(fi.sum()),'base_class_rates':{'down':float(base_prior[0]),'neutral':float(base_prior[1]),'up':float(base_prior[2])},'multiclass_logloss':ll,'base_logloss':bll,'multiclass_brier':br,'base_brier':bbr,'ece':eces,'selection_thresholds':tails,'final_tests':final,'accepted_sides':accepted,'accepted_for_display':bool(accepted),'model_trials':trials}
    return {'model':m,'calibrators':cs,'features':BASE_FEATURES,'metrics':metrics,'thresholds':{s:(tails[s]['threshold'] if tails.get(s) else None) for s in ['up','down']}},metrics


def main():
    daily=raw_daily(base.PILOT+CONTEXT);intr={s:base.fetch_15m(s) for s in base.PILOT+['SPY']};spy=spy_intraday(intr['SPY']);ctx=context_tables(daily);parts=[]
    for s in base.PILOT:
        z=stock_rows(s,intr[s],daily[s],spy,ctx)
        if not z.empty:parts.append(z)
    data=add_cross_section(pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan));bundles={};metrics={}
    for h in HORIZONS:bundles[h],metrics[str(h)]=fit_h(data,h)
    MODEL_OUT.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'version':'meaningful-move-v1','trained_at':datetime.now(timezone.utc).isoformat(),'features':BASE_FEATURES,'horizons':bundles,'definition':'volatility-adjusted meaningful up / neutral / meaningful down'},MODEL_OUT,compress=3)
    accepted=[int(h) for h,m in metrics.items() if m['accepted_for_display']];VAL_OUT.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'status':'VALIDATED' if accepted else 'NO_HORIZON_PASSED','accepted_horizons':accepted,'rows':len(data),'definition':'UP if forward return > max(0.35 * prior-day ATR% * sqrt(h), 0.6% * sqrt(h)); DOWN symmetrically; otherwise NEUTRAL. Threshold capped at 8%.','design':'2021-23 train, 2024 calibration, 2025 tail threshold/model selection, 2026+ untouched final validation','metrics':metrics},separators=(',',':')))
    print(VAL_OUT.read_text())
if __name__=='__main__':main()
