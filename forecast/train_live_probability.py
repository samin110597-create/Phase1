from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_MODEL = Path('forecast/data/live_probability_model.joblib')
OUT_VALIDATION = Path('docs/data/probability_validation.json')
TWELVE_KEY = os.getenv('TWELVE_DATA_API_KEY')
START_DATE = os.getenv('PHASE1_PROB_START', '2021-01-01')
TIMEOUT = 25
REQUESTS_PER_MINUTE = 7
ANCHOR_MINUTES = {600, 660, 720, 780, 840, 900, 945}
PILOT = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AMD','AVGO','JPM','BAC','MU']
HORIZONS = [1, 5, 10]
THRESHOLDS = [0.55, 0.575, 0.60, 0.625, 0.65, 0.675]

FEATURES = [
    'm15_ret3','m15_ret10','m15_rsi14','m15_above_ema20','m15_ema20_gt_ema50',
    'm15_macd_delta_pct','m15_atr_pct','m15_vol_ratio20','m15_vwap20_dist',
    'h4_ret','h4_ret3','h4_vol_ratio','h4_range_pct',
    'day_ret3','day_ret10','day_rsi14','day_above_ema20','day_ema20_gt_ema50',
    'day_macd_delta_pct','day_atr_pct','day_vol_ratio20','day_vwap20_dist',
    'week_ret3','week_ret10','week_rsi14','week_above_ema20','week_ema20_gt_ema50',
    'week_macd_delta_pct','week_atr_pct','week_vol_ratio20','week_vwap20_dist',
    'spy_m15_ret10','rel_m15_vs_spy','session_change','minute_norm'
]


def get_json(url: str):
    req = Request(url, headers={'User-Agent': 'Phase1-Probability-Research/2.0'})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode('utf-8'))


class RateLimiter:
    def __init__(self, limit=REQUESTS_PER_MINUTE):
        self.limit = limit; self.count = 0; self.window = time.monotonic()
    def wait(self):
        elapsed = time.monotonic() - self.window
        if elapsed >= 61:
            self.count = 0; self.window = time.monotonic(); elapsed = 0
        if self.count >= self.limit:
            sleep_for = max(0, 61 - elapsed)
            print(f'rate-limit guard: sleeping {sleep_for:.1f}s')
            time.sleep(sleep_for)
            self.count = 0; self.window = time.monotonic()
        self.count += 1


LIMITER = RateLimiter()


def fetch_15m(symbol: str) -> pd.DataFrame:
    if not TWELVE_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY missing')
    target = pd.Timestamp(START_DATE)
    end_date = None; pieces = []; seen_oldest = None
    for page in range(12):
        LIMITER.wait()
        params = {'symbol':symbol,'interval':'15min','outputsize':5000,
                  'timezone':'America/New_York','apikey':TWELVE_KEY}
        if end_date: params['end_date'] = end_date
        d = get_json('https://api.twelvedata.com/time_series?' + urlencode(params))
        vals = d.get('values') if isinstance(d, dict) else None
        if not isinstance(vals, list) or not vals:
            print(symbol, 'stopped:', d.get('message') if isinstance(d, dict) else 'no data'); break
        x = pd.DataFrame(vals)
        x['datetime'] = pd.to_datetime(x['datetime'], errors='coerce').astype('datetime64[ns]')
        for c in ['open','high','low','close','volume']:
            x[c] = pd.to_numeric(x.get(c), errors='coerce')
        x = x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime')
        if x.empty: break
        pieces.append(x[['datetime','open','high','low','close','volume']])
        oldest = x['datetime'].min(); print(symbol, 'page', page+1, 'oldest', oldest)
        if oldest <= target: break
        if seen_oldest is not None and oldest >= seen_oldest: break
        seen_oldest = oldest
        end_date = (oldest - pd.Timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%S')
    if not pieces: return pd.DataFrame()
    z = pd.concat(pieces, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
    z = z[z['datetime'] >= target]
    mins = z['datetime'].dt.hour * 60 + z['datetime'].dt.minute
    return z[(mins >= 570) & (mins <= 945)].reset_index(drop=True)


def rsi(s: pd.Series, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return 100-100/(1+rs)


def add_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x=df.copy(); c=x['close'].astype(float); v=x['volume'].fillna(0).astype(float)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); prev=c.shift(1)
    tr=pd.concat([(x['high']-x['low']),(x['high']-prev).abs(),(x['low']-prev).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean()
    typical=(x['high']+x['low']+x['close'])/3; vol20=v.rolling(20).mean(); vw20=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    x[f'{prefix}_ret3']=c.pct_change(3); x[f'{prefix}_ret10']=c.pct_change(10); x[f'{prefix}_rsi14']=rsi(c)
    x[f'{prefix}_above_ema20']=(c>e20).astype(float); x[f'{prefix}_ema20_gt_ema50']=(e20>e50).astype(float)
    x[f'{prefix}_macd_delta_pct']=(macd-sig)/c; x[f'{prefix}_atr_pct']=atr/c; x[f'{prefix}_vol_ratio20']=v/vol20.replace(0,np.nan); x[f'{prefix}_vwap20_dist']=c/vw20-1
    return x


def daily_history(symbols: list[str]) -> dict[str,pd.DataFrame]:
    raw=yf.download(symbols,start='2019-01-01',auto_adjust=True,actions=False,group_by='ticker',threads=True,progress=False)
    out={}
    for s in symbols:
        try:
            d=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy(); d=d.rename(columns=str.lower).dropna(subset=['close'])
            d.index=pd.to_datetime(d.index).tz_localize(None).astype('datetime64[ns]'); out[s]=d[['open','high','low','close','volume']].copy()
        except Exception: out[s]=pd.DataFrame()
    return out


def daily_feature_table(d):
    x=d.reset_index().rename(columns={d.index.name or 'index':'datetime'}); x['datetime']=pd.to_datetime(x['datetime']).astype('datetime64[ns]')
    f=add_features(x,'day').set_index(x['datetime'].dt.normalize()); cols=[c for c in f.columns if c.startswith('day_')]
    return f[cols].shift(1)


def weekly_available_table(d):
    w=pd.DataFrame({'open':d['open'].resample('W-FRI').first(),'high':d['high'].resample('W-FRI').max(),'low':d['low'].resample('W-FRI').min(),'close':d['close'].resample('W-FRI').last(),'volume':d['volume'].resample('W-FRI').sum()}).dropna(subset=['close'])
    z=add_features(w.reset_index().rename(columns={w.index.name or 'index':'datetime'}),'week')
    z['available_date']=pd.to_datetime(z['datetime']).astype('datetime64[ns]')+pd.Timedelta(days=3)
    return z[['available_date']+[c for c in z.columns if c.startswith('week_')]].sort_values('available_date')


def make_symbol_rows(symbol,intraday,daily,spy_intraday_features):
    if intraday.empty or daily.empty: return pd.DataFrame()
    z=add_features(intraday,'m15'); c=z['close']; v=z['volume'].fillna(0)
    z['h4_ret']=c.pct_change(16); z['h4_ret3']=c.pct_change(48); z['h4_vol_ratio']=v.rolling(16).mean()/v.rolling(80).mean().replace(0,np.nan)
    z['h4_range_pct']=(z['high'].rolling(16).max()-z['low'].rolling(16).min())/c
    z['snapshot_date']=z['datetime'].dt.normalize().astype('datetime64[ns]'); z['minute']=z['datetime'].dt.hour*60+z['datetime'].dt.minute
    z=z[z['minute'].isin(ANCHOR_MINUTES)].copy()
    if z.empty: return z
    d=daily.copy(); d.index=pd.to_datetime(d.index).astype('datetime64[ns]').normalize()
    z=z.join(daily_feature_table(d),on='snapshot_date')
    z=pd.merge_asof(z.sort_values('snapshot_date'),weekly_available_table(d),left_on='snapshot_date',right_on='available_date',direction='backward')
    prev_close=d['close'].shift(1); z['prev_close']=z['snapshot_date'].map(prev_close); z['session_change']=z['close']/z['prev_close']-1; z['minute_norm']=(z['minute']-570)/390.0
    z=z.merge(spy_intraday_features[['datetime','spy_m15_ret10']],on='datetime',how='left'); z['rel_m15_vs_spy']=z['m15_ret10']-z['spy_m15_ret10']
    z['same_day_daily_close']=z['snapshot_date'].map(d['close']); scale_gap=(z['close']/z['same_day_daily_close']-1).abs(); z=z[(scale_gap.isna())|(scale_gap<0.08)]
    for h in HORIZONS:
        future_close=d['close'].shift(-h); z[f'future_close_{h}']=z['snapshot_date'].map(future_close); z[f'fwd_{h}d_return']=z[f'future_close_{h}']/z['close']-1
        z[f'y_up_{h}']=(z[f'fwd_{h}d_return']>0).astype(float); z.loc[z[f'fwd_{h}d_return'].isna(),f'y_up_{h}']=np.nan
    z['symbol']=symbol
    return z


def safe_logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5); return np.log(p/(1-p)).reshape(-1,1)


def wilson_lower(success_rate,n,z=1.96):
    if n<=0: return 0.0
    p=float(success_rate); den=1+z*z/n; centre=p+z*z/(2*n); spread=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)
    return max(0.0,(centre-spread)/den)


def ece_and_bins(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); bins=[]; ece=0.0; edges=np.linspace(0,1,11)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi)); n=int(m.sum())
        if not n: continue
        mp=float(p[m].mean()); obs=float(y[m].mean()); ece+=(n/len(y))*abs(mp-obs)
        bins.append({'lo':round(float(lo),3),'hi':round(float(hi),3),'n':n,'mean_pred':round(mp,4),'observed_up':round(obs,4),'wilson_lower':round(wilson_lower(obs,n),4)})
    return float(ece),bins


def side_arrays(y,p,side):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return (y,p) if side=='up' else (1-y,1-p)


def choose_threshold(y,p,side):
    sy,sp=side_arrays(y,p,side); base=float(sy.mean()); choices=[]
    for t in THRESHOLDS:
        m=sp>=t; n=int(m.sum())
        if n<200: continue
        obs=float(sy[m].mean()); pred=float(sp[m].mean()); low=wilson_lower(obs,n); gap=abs(obs-pred)
        choices.append({'threshold':t,'n':n,'observed':obs,'mean_pred':pred,'wilson_lower':low,'calibration_gap':gap,'base_rate':base})
    eligible=[x for x in choices if x['observed']>=0.58 and x['wilson_lower']>base+0.01 and x['calibration_gap']<=0.06]
    chosen=max(eligible,key=lambda x:(x['wilson_lower'],x['n'])) if eligible else None
    return chosen,choices


def threshold_eval(y,p,side,threshold):
    sy,sp=side_arrays(y,p,side); base=float(sy.mean()); m=sp>=threshold; n=int(m.sum())
    if not n: return {'n':0,'base_rate':round(base,4),'accepted':False}
    obs=float(sy[m].mean()); pred=float(sp[m].mean()); low=wilson_lower(obs,n); gap=abs(obs-pred)
    accepted=bool(n>=80 and obs>=0.58 and low>base and gap<=0.06)
    return {'n':n,'base_rate':round(base,4),'mean_probability':round(pred,4),'observed_rate':round(obs,4),'wilson_lower_95':round(low,4),'calibration_gap':round(gap,4),'accepted':accepted}


def precision_at_k(frame,pcol,labelcol,k,side,threshold=None):
    total=[]; snapshots=0
    for _,g in frame.groupby('datetime',sort=False):
        gg=g.copy()
        if side=='up': gg['side_p']=gg[pcol]; gg['side_y']=gg[labelcol]
        else: gg['side_p']=1-gg[pcol]; gg['side_y']=1-gg[labelcol]
        if threshold is not None: gg=gg[gg['side_p']>=threshold]
        if gg.empty: continue
        take=gg.nlargest(min(k,len(gg)),'side_p'); total.extend(take['side_y'].astype(float).tolist()); snapshots+=1
    if not total: return {'picks':0,'snapshots':0,'precision':None,'wilson_lower_95':None}
    obs=float(np.mean(total)); return {'picks':len(total),'snapshots':snapshots,'precision':round(obs,4),'wilson_lower_95':round(wilson_lower(obs,len(total)),4)}


def fit_horizon(data,h):
    target=f'y_up_{h}'; valid=data[target].notna(); d=data.loc[valid].copy().reset_index(drop=True)
    X=d[FEATURES].replace([np.inf,-np.inf],np.nan); y=d[target].astype(int); dates=pd.to_datetime(d['snapshot_date'])
    train=dates<pd.Timestamp('2024-01-01'); cal=(dates>=pd.Timestamp('2024-01-01'))&(dates<pd.Timestamp('2025-01-01'))
    select=(dates>=pd.Timestamp('2025-01-01'))&(dates<pd.Timestamp('2026-01-01')); final=dates>=pd.Timestamp('2026-01-01')
    Xtr,ytr=X[train],y[train]; Xc,yc=X[cal],y[cal]; Xs,ys=X[select],y[select]; Xf,yf_=X[final],y[final]
    if min(len(Xtr),len(Xc),len(Xs),len(Xf))<1000: raise RuntimeError(f'horizon {h}: insufficient chronological samples')
    candidates={
      'logistic':Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=0.5,max_iter=1200,random_state=42))]),
      'hist_gradient_boosting':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(max_depth=4,learning_rate=0.05,max_iter=180,l2_regularization=1.0,random_state=42))])}
    chosen_name=None; chosen=None; best=math.inf
    for name,model in candidates.items():
        model.fit(Xtr,ytr); pc=np.clip(model.predict_proba(Xc)[:,1],1e-6,1-1e-6); loss=log_loss(yc,pc); print('horizon',h,name,'cal logloss',loss)
        if loss<best: best=loss; chosen_name=name; chosen=model
    rawc=np.clip(chosen.predict_proba(Xc)[:,1],1e-5,1-1e-5); calibrator=LogisticRegression(C=1e6,max_iter=500,random_state=42); calibrator.fit(safe_logit(rawc),yc)
    ps=calibrator.predict_proba(safe_logit(np.clip(chosen.predict_proba(Xs)[:,1],1e-5,1-1e-5)))[:,1]
    pf=calibrator.predict_proba(safe_logit(np.clip(chosen.predict_proba(Xf)[:,1],1e-5,1-1e-5)))[:,1]
    base=float(ytr.mean()); basef=np.full(len(yf_),base)
    brier=float(brier_score_loss(yf_,pf)); bb=float(brier_score_loss(yf_,basef)); ll=float(log_loss(yf_,np.clip(pf,1e-6,1-1e-6))); bll=float(log_loss(yf_,np.clip(basef,1e-6,1-1e-6)))
    auc=float(roc_auc_score(yf_,pf)) if len(np.unique(yf_))>1 else None; ece,bins=ece_and_bins(yf_,pf); overall_ok=bool(brier<bb and ll<bll and ece<=0.05)
    up_sel,up_choices=choose_threshold(ys,ps,'up'); dn_sel,dn_choices=choose_threshold(ys,ps,'down')
    up_final=threshold_eval(yf_,pf,'up',up_sel['threshold']) if up_sel else {'accepted':False,'n':0}; dn_final=threshold_eval(yf_,pf,'down',dn_sel['threshold']) if dn_sel else {'accepted':False,'n':0}
    ff=d.loc[final,['datetime',target]].copy(); ff['p']=pf
    for side,selx,finx in [('up',up_sel,up_final),('down',dn_sel,dn_final)]:
        if selx:
            finx['precision_at_1']=precision_at_k(ff,'p',target,1,side,selx['threshold']); finx['precision_at_3']=precision_at_k(ff,'p',target,3,side,selx['threshold']); finx['precision_at_5']=precision_at_k(ff,'p',target,5,side,selx['threshold'])
            p3=finx['precision_at_3']; side_base=finx.get('base_rate',0)
            finx['accepted']=bool(overall_ok and finx.get('accepted') and p3.get('picks',0)>=80 and p3.get('precision') is not None and p3['precision']>=0.58 and p3['wilson_lower_95']>side_base)
    accepted_sides=[s for s,x in [('up',up_final),('down',dn_final)] if x.get('accepted')]
    metrics={'horizon_sessions':h,'model':chosen_name,'train_n':int(len(ytr)),'calibration_n':int(len(yc)),'selection_2025_n':int(len(ys)),'final_2026_n':int(len(yf_)),
             'final_start':'2026-01-01','final_end':str(dates[final].max().date()),'base_up_rate_train':round(base,4),'final_observed_up_rate':round(float(yf_.mean()),4),
             'brier':round(brier,5),'base_brier':round(bb,5),'log_loss':round(ll,5),'base_log_loss':round(bll,5),'roc_auc':round(auc,4) if auc is not None else None,'ece_10bin':round(ece,4),
             'overall_probability_quality_pass':overall_ok,'accepted_sides':accepted_sides,'accepted_for_display':bool(accepted_sides),'selection_thresholds':{'up':up_sel,'down':dn_sel},
             'selection_candidates':{'up':up_choices,'down':dn_choices},'final_high_probability_tests':{'up':up_final,'down':dn_final},'calibration_bins_final':bins}
    bundle={'model':chosen,'calibrator':calibrator,'features':FEATURES,'metrics':metrics,'display_thresholds':{'up':up_sel['threshold'] if up_sel else None,'down':dn_sel['threshold'] if dn_sel else None}}
    return bundle,metrics


def main():
    if not TWELVE_KEY: raise RuntimeError('TWELVE_DATA_API_KEY must exist in GitHub Secrets')
    daily=daily_history(PILOT+['SPY']); intraday={}
    for s in PILOT+['SPY']:
        intraday[s]=fetch_15m(s); print(s,'15m rows',len(intraday[s]))
    spy=add_features(intraday['SPY'].copy(),'spy_m15'); spy_feats=spy[['datetime','spy_m15_ret10']].copy()
    parts=[]; coverage={}
    for s in PILOT:
        z=make_symbol_rows(s,intraday.get(s,pd.DataFrame()),daily.get(s,pd.DataFrame()),spy_feats)
        if not z.empty:
            parts.append(z); coverage[s]={'rows':int(len(z)),'first':str(z['datetime'].min()),'last':str(z['datetime'].max())}
    if not parts: raise RuntimeError('no training rows built')
    data=pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan); print('combined rows',len(data))
    bundles={}; metrics={}
    for h in HORIZONS:
        bundle,m=fit_horizon(data,h); bundles[h]=bundle; metrics[str(h)]=m
    OUT_MODEL.parent.mkdir(parents=True,exist_ok=True)
    artifact={'version':'live-probability-v2-strict','trained_at':datetime.now(timezone.utc).isoformat(),'feature_names':FEATURES,'horizons':bundles,'definition':'P(next h-session close is above current intraday snapshot price)'}
    joblib.dump(artifact,OUT_MODEL,compress=3)
    accepted=[int(h) for h,m in metrics.items() if m['accepted_for_display']]
    validation={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'STRICT_VALIDATED_V2' if accepted else 'NO_HORIZON_PASSED_STRICT_VALIDATION',
      'probability_definition':'P(stock price is higher at the close h trading sessions ahead than at the current intraday snapshot)',
      'training_design':'2021-2023 model training; 2024 probability calibration; 2025 high-probability threshold selection; 2026+ final untouched validation.',
      'calibration_method':'sigmoid/Platt calibration fitted only on 2024 predictions',
      'acceptance_rule':'Final 2026+ overall Brier/log loss must beat the train-base-rate predictor, ECE <=5%, and a 2025-frozen high-probability side threshold must have >=80 final observations, >=58% observed success, Wilson 95% lower bound above the final side base rate, <=6% calibration gap, and selective Precision@3 must also pass.',
      'accepted_horizons':accepted,'features':FEATURES,'symbols':PILOT,'rows_total':int(len(data)),'coverage':coverage,'metrics':metrics,
      'limitations':['current-survivor pilot universe','12-stock pilot only','historical provider coverage may vary','probabilities are empirical estimates, not guarantees']}
    OUT_VALIDATION.parent.mkdir(parents=True,exist_ok=True); OUT_VALIDATION.write_text(json.dumps(validation,separators=(',',':'))); print(json.dumps(validation,indent=2))


if __name__=='__main__': main()
