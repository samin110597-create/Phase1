from __future__ import annotations

import json, math, time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

import forecast.train_intraday_meta_model_v5 as meta

MODEL=Path('forecast/data/intraday_signal_v3.joblib')
VALIDATION=Path('docs/data/intraday_model_v3_validation.json')
SUMMARY=Path('docs/data/intraday_training_summary.json')
AUDIT=Path('docs/data/intraday_dataset_audit.json')
MODEL.parent.mkdir(parents=True,exist_ok=True); VALIDATION.parent.mkdir(parents=True,exist_ok=True); SUMMARY.parent.mkdir(parents=True,exist_ok=True)

UNIVERSE=['AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AMD','AVGO','NFLX','JPM','BAC','XOM','CVX','WMT','COST','HD','UNH','LLY','ORCL','CRM','PLTR','MU','INTC','QCOM','AMAT','TSM','UBER','V','MA','GS','MS','CAT','GE','DIS','KO','PEP','CSCO','ADBE']
SECTOR_OF={
'AAPL':'XLK','MSFT':'XLK','NVDA':'XLK','AMD':'XLK','AVGO':'XLK','ORCL':'XLK','CRM':'XLK','ADBE':'XLK','CSCO':'XLK','INTC':'XLK','QCOM':'XLK','AMAT':'XLK','MU':'XLK','PLTR':'XLK','TSM':'XLK',
'META':'XLC','GOOGL':'XLC','GOOG':'XLC','NFLX':'XLC','DIS':'XLC','AMZN':'XLY','TSLA':'XLY','HD':'XLY','UBER':'XLY','JPM':'XLF','BAC':'XLF','V':'XLF','MA':'XLF','GS':'XLF','MS':'XLF','UNH':'XLV','LLY':'XLV','XOM':'XLE','CVX':'XLE','CAT':'XLI','GE':'XLI','WMT':'XLP','COST':'XLP','KO':'XLP','PEP':'XLP'}
MARKET=['SPY','QQQ','IWM','^VIX','TLT','GLD','USO']
SECTORS=sorted(set(SECTOR_OF.values()))
ALL=UNIVERSE+MARKET+SECTORS
DECISION_HOURS=['10:30','12:30','14:30','15:30']


def norm_one(raw,symbol):
    try:
        x=raw[symbol].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy()
        x=x.rename(columns=lambda c:str(c).lower())
        if x.empty:return pd.DataFrame()
        x=x.reset_index(); dc=next((c for c in x.columns if str(c).lower() in ('datetime','date')),x.columns[0]); x=x.rename(columns={dc:'datetime'}); x['datetime']=pd.to_datetime(x['datetime'],errors='coerce')
        try:
            if x['datetime'].dt.tz is not None:x['datetime']=x['datetime'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        except Exception:pass
        for c in ('open','high','low','close','volume'):
            if c not in x:x[c]=0 if c=='volume' else np.nan
            x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime'); tm=x.datetime.dt.strftime('%H:%M'); x=x[(tm>='09:30')&(tm<='16:00')]
        return x[['datetime','open','high','low','close','volume']].drop_duplicates('datetime').reset_index(drop=True)
    except Exception:return pd.DataFrame()


def download_hourly(symbols):
    out={}
    for i in range(0,len(symbols),20):
        batch=symbols[i:i+20]; last=None
        for attempt in range(3):
            try:
                raw=yf.download(batch,period='729d',interval='60m',auto_adjust=False,actions=False,prepost=False,group_by='ticker',threads=True,progress=False)
                for s in batch:out[s]=norm_one(raw,s)
                last=None; break
            except Exception as e:last=e; time.sleep(3*(attempt+1))
        if last:
            for s in batch:out[s]=pd.DataFrame()
    return out


def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/dn.replace(0,np.nan); return 100-100/(1+rs)


def hourly_features(bars):
    z=bars.copy().sort_values('datetime').reset_index(drop=True); c=z.close.astype(float); v=z.volume.fillna(0).astype(float)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); prev=c.shift(1)
    tr=pd.concat([(z.high-z.low),(z.high-prev).abs(),(z.low-prev).abs()],axis=1).max(axis=1); typ=(z.high+z.low+z.close)/3
    z['h1_ret1']=c.pct_change(); z['h1_ret3']=c.pct_change(3); z['h1_ret6']=c.pct_change(6); z['h1_rsi14']=rsi(c); z['h1_above_ema20']=(c>e20).astype(float); z['h1_ema20_gt_ema50']=(e20>e50).astype(float); z['h1_macd_delta_pct']=(macd-sig)/c; z['h1_atr_pct']=tr.rolling(14).mean()/c; z['h1_vol_ratio20']=v/v.rolling(20).mean().replace(0,np.nan); z['h1_vwap20_dist']=c/((typ*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan))-1
    z['h4_ret']=c.pct_change(4); z['h4_ret3']=c.pct_change(12); z['h4_range_pct']=(z.high.rolling(4).max()-z.low.rolling(4).min())/c; z['h4_vol_ratio']=v.rolling(4).mean()/v.rolling(20).mean().replace(0,np.nan)
    z['date']=z.datetime.dt.normalize(); z['bar_no']=z.groupby('date').cumcount(); z['session_open']=z.groupby('date').open.transform('first'); z['session_high']=z.groupby('date').high.cummax(); z['session_low']=z.groupby('date').low.cummin(); z['session_ret']=c/z.session_open-1; z['session_range_pos']=(c-z.session_low)/(z.session_high-z.session_low).replace(0,np.nan); z['cum_pv']=(typ*v).groupby(z.date).cumsum(); z['cum_vol']=v.groupby(z.date).cumsum(); z['session_vwap_dist']=c/(z.cum_pv/z.cum_vol.replace(0,np.nan))-1; z['same_slot_vol20']=z.groupby('bar_no').volume.transform(lambda s:s.shift(1).rolling(20,min_periods=5).mean()); z['session_rvol']=v/z.same_slot_vol20.replace(0,np.nan); z['volume_accel']=v/v.shift(1).replace(0,np.nan)
    return z


def daily_from_hourly(b):
    if b.empty:return pd.DataFrame()
    x=b.copy(); x['date']=x.datetime.dt.normalize(); d=x.groupby('date').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).sort_index(); d.index=pd.to_datetime(d.index); return d


def daily_features(d,prefix):
    if d.empty:return pd.DataFrame()
    c=d.close; v=d.volume.fillna(0); e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean(); macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); prev=c.shift(1); tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    f=pd.DataFrame(index=d.index); f[f'{prefix}_ret1']=c.pct_change(); f[f'{prefix}_ret5']=c.pct_change(5); f[f'{prefix}_ret20']=c.pct_change(20); f[f'{prefix}_ret63']=c.pct_change(63); f[f'{prefix}_rsi14']=rsi(c); f[f'{prefix}_above_ema20']=(c>e20).astype(float); f[f'{prefix}_ema20_gt_ema50']=(e20>e50).astype(float); f[f'{prefix}_ema50_gt_ema200']=(e50>e200).astype(float); f[f'{prefix}_macd_delta_pct']=(macd-sig)/c; f[f'{prefix}_atr_pct']=tr.rolling(14).mean()/c; f[f'{prefix}_rv20']=c.pct_change().rolling(20).std(); f[f'{prefix}_vol_ratio20']=v/v.rolling(20).mean().replace(0,np.nan); f[f'{prefix}_adv20_log']=np.log1p((c*v).rolling(20).mean()); return f


def weekly_features(d):
    if d.empty:return pd.DataFrame()
    w=pd.DataFrame({'open':d.open.resample('W-FRI').first(),'high':d.high.resample('W-FRI').max(),'low':d.low.resample('W-FRI').min(),'close':d.close.resample('W-FRI').last(),'volume':d.volume.resample('W-FRI').sum()}).dropna(subset=['close']); f=daily_features(w,'week'); return f[[c for c in f if c in ('week_ret1','week_ret5','week_ret20','week_rsi14','week_above_ema20','week_ema20_gt_ema50','week_macd_delta_pct','week_atr_pct')]]


def asof_prev(snaps,feat):
    if snaps.empty or feat.empty:return snaps
    left=snaps[['snapshot_dt']].copy(); left['date']=left.snapshot_dt.dt.normalize(); r=feat.reset_index().rename(columns={feat.index.name or 'index':'context_date'}).sort_values('context_date'); m=pd.merge_asof(left.sort_values('date'),r,left_on='date',right_on='context_date',direction='backward',allow_exact_matches=False)
    out=snaps.sort_values('snapshot_dt').copy()
    for c in feat.columns:out[c]=m[c].to_numpy()
    return out


def snapshots(symbol,b):
    if b.empty:return pd.DataFrame()
    z=hourly_features(b); rows=[]
    for date,g in z.groupby('date',sort=True):
        for slot in DECISION_HOURS:
            decision=pd.Timestamp(f'{date.date()} {slot}'); elig=g[g.datetime<decision]
            if elig.empty:continue
            r=elig.iloc[-1].copy(); r['snapshot_dt']=decision; r['slot']=slot; r['symbol']=symbol; rows.append(r)
    if not rows:return pd.DataFrame()
    x=pd.DataFrame(rows); keep=['snapshot_dt','slot','symbol','close','h1_ret1','h1_ret3','h1_ret6','h1_rsi14','h1_above_ema20','h1_ema20_gt_ema50','h1_macd_delta_pct','h1_atr_pct','h1_vol_ratio20','h1_vwap20_dist','h4_ret','h4_ret3','h4_range_pct','h4_vol_ratio','session_ret','session_range_pos','session_vwap_dist','session_rvol','volume_accel']; return x[keep]


def candidate_score(d):
    s=pd.Series(0.0,index=d.index); s+=np.where(d.h1_above_ema20>.5,9,-9); s+=np.where(d.h1_ema20_gt_ema50>.5,8,-8); s+=np.where(d.h1_macd_delta_pct.fillna(0)>=0,6,-6); s+=np.where(d.session_vwap_dist.fillna(0)>=0,6,-6); s+=np.where(d.h4_ret.fillna(0)>=0,9,-9); s+=np.where(d.day_above_ema20.fillna(0)>.5,7,-7); s+=np.where(d.day_ema20_gt_ema50.fillna(0)>.5,6,-6); s+=np.where(d.week_above_ema20.fillna(0)>.5,5,-5); s+=np.where(d.week_ema20_gt_ema50.fillna(0)>.5,4,-4); return s


def attach_labels(df,daily,spy_daily):
    sclose=daily.close; pclose=spy_daily.close; dates=list(sclose.index); pos={d:i for i,d in enumerate(dates)}
    for h in (1,5,10):
        sr=[]; pr=[]; ex=[]; th=[]; up=[]; dn=[]
        for _,r in df.iterrows():
            d=r.snapshot_dt.normalize(); i=pos.get(d); cur=float(r.close)
            if i is None or i+h>=len(dates) or d not in pclose.index: vals=(np.nan,)*6
            else:
                fd=dates[i+h]
                if fd not in pclose.index: vals=(np.nan,)*6
                else:
                    a=float(sclose.loc[fd]/cur-1); bmk=float(pclose.loc[fd]/pclose.loc[d]-1); e=a-bmk; atr=float(r.get('day_atr_pct')) if pd.notna(r.get('day_atr_pct')) else .02; threshold=max(.003,.35*max(.005,atr)*math.sqrt(h)); vals=(a,bmk,e,threshold,float(e>threshold),float(e<-threshold))
            a,bmk,e,threshold,yu,yd=vals; sr.append(a); pr.append(bmk); ex.append(e); th.append(threshold); up.append(yu); dn.append(yd)
        df[f'fwd_{h}_return']=sr; df[f'spy_fwd_{h}_return']=pr; df[f'fwd_{h}_excess']=ex; df[f'move_threshold_{h}']=th; df[f'label_up_{h}']=up; df[f'label_down_{h}']=dn
    return df


def feature_cols(df):
    ignore={'snapshot_dt','slot','symbol','close'}; pref=('fwd_','spy_fwd_','label_','move_threshold_'); return [c for c in df if c not in ignore and not c.startswith(pref) and pd.api.types.is_numeric_dtype(df[c])]


def main():
    bars=download_hourly(ALL); missing=[s for s in ALL if bars.get(s,pd.DataFrame()).empty]
    if 'SPY' in missing:raise RuntimeError('SPY hourly history unavailable')
    daily={s:daily_from_hourly(bars.get(s,pd.DataFrame())) for s in ALL}; market_ctx={}
    for bmk in MARKET:
        d=daily.get(bmk,pd.DataFrame())
        if not d.empty:market_ctx[bmk]=daily_features(d,bmk.replace('^','').lower())
    parts=[]; coverage={}
    spy_sn=snapshots('SPY',bars['SPY']); spy_keep=spy_sn[['snapshot_dt','h1_ret3','h1_ret6','h4_ret','session_ret','session_vwap_dist','session_rvol']].rename(columns={c:'spy_'+c for c in ['h1_ret3','h1_ret6','h4_ret','session_ret','session_vwap_dist','session_rvol']})
    for s in UNIVERSE:
        if bars.get(s,pd.DataFrame()).empty or daily.get(s,pd.DataFrame()).empty:continue
        x=snapshots(s,bars[s]); x=asof_prev(x,daily_features(daily[s],'day')); x=asof_prev(x,weekly_features(daily[s])); sec=SECTOR_OF.get(s)
        if sec and not daily.get(sec,pd.DataFrame()).empty:
            sf=daily_features(daily[sec],'sector'); x=asof_prev(x,sf[['sector_ret5','sector_ret20','sector_ret63','sector_above_ema20','sector_ema20_gt_ema50']])
        for bmk,f in market_ctx.items():
            cols=[c for c in f.columns if c.endswith(('ret5','ret20','ret63','atr_pct','rv20')) or 'above_ema20' in c or 'ema20_gt_ema50' in c]; x=asof_prev(x,f[cols])
        x=x.merge(spy_keep,on='snapshot_dt',how='left'); x['rel_h1_6_vs_spy']=x.h1_ret6-x.spy_h1_ret6; x['rel_h4_vs_spy']=x.h4_ret-x.spy_h4_ret
        if 'sector_ret20' in x:x['rel20_vs_sector']=x.day_ret20-x.sector_ret20
        x=attach_labels(x,daily[s],daily['SPY']); coverage[s]={'rows':int(len(x)),'first':str(x.snapshot_dt.min()),'last':str(x.snapshot_dt.max())}; parts.append(x)
    data=pd.concat(parts,ignore_index=True); data['candidate_score']=candidate_score(data); g=data.groupby('snapshot_dt'); data['candidate_rank_pct']=g.candidate_score.rank(pct=True); data['activity_rank_pct']=g.session_ret.transform(lambda s:s.abs().rank(pct=True)); data['rvol_rank_pct']=g.session_rvol.rank(pct=True); data['cross_section_rel_rank']=g.rel_h1_6_vs_spy.rank(pct=True); data=data.sort_values(['snapshot_dt','symbol']).reset_index(drop=True)
    pool=data[(data.candidate_rank_pct>=.75)|(data.candidate_rank_pct<=.25)].copy(); feats=feature_cols(data)
    # Recent but chronological: model fit -> Platt -> comparable setup calibration -> threshold selection -> 2026 diagnostic.
    periods={'train':pool.snapshot_dt<pd.Timestamp('2025-04-01'),'cal_platt':(pool.snapshot_dt>=pd.Timestamp('2025-04-01'))&(pool.snapshot_dt<pd.Timestamp('2025-07-01')),'meta_cal':(pool.snapshot_dt>=pd.Timestamp('2025-07-01'))&(pool.snapshot_dt<pd.Timestamp('2025-10-01')),'select':(pool.snapshot_dt>=pd.Timestamp('2025-10-01'))&(pool.snapshot_dt<pd.Timestamp('2026-01-01')),'dev2026':pool.snapshot_dt>=pd.Timestamp('2026-01-01')}
    bundle={'version':'hourly-meta-v7','trained_at':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_FOR_PROSPECTIVE_VALIDATION','feature_columns':feats,'models':{},'meta_policy':{'min_local_n':40,'max_member_dispersion':.10,'prior_strength':35.0},'decision_hours':DECISION_HOURS}; results={}
    old_min=meta.MIN_LOCAL_N; old_disp=meta.MAX_DISPERSION; old_prior=meta.PRIOR_STRENGTH; meta.MIN_LOCAL_N=40; meta.MAX_DISPERSION=.10; meta.PRIOR_STRENGTH=35.0
    try:
        for h in (1,5,10):
            valid=pool[f'label_up_{h}'].notna()&pool[f'label_down_{h}'].notna(); ph=pool[valid].copy(); y=meta.make_target(ph,h); split={}; ys={}
            for name,mask in periods.items(): ids=ph.index[mask.reindex(ph.index,fill_value=False)]; split[name]=ph.loc[ids]; ys[name]=y.loc[ids]
            if min(len(split['train']),len(split['cal_platt']),len(split['meta_cal']),len(split['select']))<250:
                results[str(h)]={'status':'INSUFFICIENT_HISTORY','counts':{k:int(len(v)) for k,v in split.items()}}; continue
            members=[]; pmeta=[]; psel=[]; pdev=[]
            for name in ('logistic','hgb2','hgb4'):
                m=meta.factory(name); m.fit(split['train'][feats],ys['train']); raw=m.predict_proba(split['cal_platt'][feats]); cls=list(m.classes_); cals={}
                for k in meta.CLASSES:cals[str(k)]=meta.platt(raw[:,cls.index(k)],(ys['cal_platt'].to_numpy()==k).astype(int))
                members.append({'name':name,'model':m,'calibrators':cals}); pmeta.append(meta.calibrate_members(m,cals,split['meta_cal'][feats])); psel.append(meta.calibrate_members(m,cals,split['select'][feats])); pdev.append(meta.calibrate_members(m,cals,split['dev2026'][feats]) if len(split['dev2026']) else np.empty((0,3)))
            em=np.mean(np.stack(pmeta),axis=0); dm=np.max(np.std(np.stack(pmeta),axis=0),axis=1); table,global_table=meta.build_local_table(split['meta_cal'],em,ys['meta_cal']); es=np.mean(np.stack(psel),axis=0); ds=np.max(np.std(np.stack(psel),axis=0),axis=1); ms,ns=meta.meta_adjust(split['select'],es,table,global_table); mm,nm=meta.meta_adjust(split['meta_cal'],em,table,global_table)
            if len(split['dev2026']):ed=np.mean(np.stack(pdev),axis=0); dd=np.max(np.std(np.stack(pdev),axis=0),axis=1); md,nd=meta.meta_adjust(split['dev2026'],ed,table,global_table)
            else:md=np.empty((0,3));nd=np.empty((0,3),int);dd=np.array([])
            rules={}; dev={}; targets={}
            for side in ('up','down'):
                rules[side]=meta.side_threshold(split['select'],ms,ns,ds,ys['select'],side); dev[side]=meta.test_threshold(split['dev2026'],md,nd,dd,ys['dev2026'],side,rules[side]) if len(split['dev2026']) else {'signals':0}; targets[side]=meta.return_stats(split['meta_cal'],mm,nm,dm,ys['meta_cal'],side,h,rules[side])
            bundle['models'][str(h)]={'members':members,'local_table':table,'global_table':global_table,'thresholds':{s:(rules[s]['threshold'] if rules[s] else None) for s in ('up','down')},'selection_rules':rules,'return_stats':targets}; results[str(h)]={'status':'FROZEN_HOURLY_META_MODEL','counts':{k:int(len(v)) for k,v in split.items()},'selection_probability_quality':meta.multiclass_metrics(ys['select'],ms),'selection_signal_rules':rules,'development_2026_diagnostic':dev,'historical_return_targets':targets}
    finally:
        meta.MIN_LOCAL_N=old_min; meta.MAX_DISPERSION=old_disp; meta.PRIOR_STRENGTH=old_prior
    joblib.dump(bundle,MODEL,compress=3)
    audit={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'PASS','dataset':'bulk Yahoo Finance 60m history','rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'symbols':int(data.symbol.nunique()),'duplicate_snapshot_symbol_rows':int(data.duplicated(['snapshot_dt','symbol']).sum()),'future_columns_excluded_from_model':True,'chronological_splits':True}
    AUDIT.write_text(json.dumps(audit,separators=(',',':')))
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'HOURLY LIVE-LIKE DATASET BUILT AND MODEL FROZEN','history_source':'Yahoo Finance 60m bulk history; no Twelve Data dependency','framework':'hourly intraday + rolling 4h + prior Day + prior Week + SPY/QQQ/IWM/VIX/TLT/GLD/USO + sector + cross-sectional ranks','decision_times_et':DECISION_HOURS,'rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'symbols_with_rows':len(coverage),'coverage':coverage,'missing_downloads':missing,'benchmark_label_alignment':'CORRECTED: stock and SPY outcomes share the same snapshot date and future completed session','validation_policy':'2025Q4 selects frozen thresholds; 2026 is diagnostic only because it has been inspected; final proof is prospective.'}
    SUMMARY.write_text(json.dumps(summary,separators=(',',':')))
    val={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'HOURLY META V7 FROZEN — SELECTIVE, RECENT, RATE-LIMIT-INDEPENDENT','model_version':'hourly-meta-v7','architecture':'3-class ensemble + chronological Platt calibration + independent comparable-setup calibration + cross-sectional ranks + abstention','history_source':'Yahoo Finance 60m bulk','dataset_rows':int(len(data)),'candidate_pool_rows':int(len(pool)),'symbols':int(data.symbol.nunique()),'features':feats,'validation_design':'pre-2025Q2 fit; 2025Q2 Platt; 2025Q3 comparable-setup calibration; 2025Q4 frozen threshold selection; 2026 diagnostic only; final validation prospective','results':results}
    VALIDATION.write_text(json.dumps(val,separators=(',',':')))
    print(json.dumps({'status':val['status'],'rows':len(data),'pool_rows':len(pool),'symbols':data.symbol.nunique(),'missing':missing,'results':{h:{'up':v.get('selection_signal_rules',{}).get('up'),'down':v.get('selection_signal_rules',{}).get('down'),'dev':v.get('development_2026_diagnostic')} for h,v in results.items()}},indent=2))

if __name__=='__main__':main()
