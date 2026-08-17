from __future__ import annotations

import json, math, sys
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd, yfinance as yf

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v4 as v4
import scripts.github_live_forecast_v7 as prev
import forecast.build_intraday_live_like_dataset as hist
from forecast.live_intraday_v3_inference import available as model_available, predict

OUT=Path('docs/data/live_forecast.json')
MODEL=Path('forecast/data/intraday_signal_v3.joblib')


def meta_ready():
    if not model_available() or not MODEL.exists(): return False, None
    try:
        ver=joblib.load(MODEL).get('version')
        return ver in ('intraday-meta-v5','intraday-meta-v6'), ver
    except Exception:
        return False, None


def bulk_15m(symbols):
    raw=yf.download(sorted(set(symbols)),period='10d',interval='15m',auto_adjust=False,actions=False,prepost=False,group_by='ticker',threads=True,progress=False)
    out={}
    for s in sorted(set(symbols)):
        try:
            x=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy(); x=x.rename(columns=lambda c:str(c).lower())
            x=x.reset_index(); dc=next((c for c in x.columns if str(c).lower() in ('datetime','date')),x.columns[0]); x=x.rename(columns={dc:'datetime'}); x['datetime']=pd.to_datetime(x['datetime'],errors='coerce')
            try:
                if x['datetime'].dt.tz is not None:x['datetime']=x['datetime'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            except Exception:pass
            for c in ('open','high','low','close','volume'):x[c]=pd.to_numeric(x.get(c),errors='coerce')
            x=x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime'); tm=x.datetime.dt.strftime('%H:%M'); x=x[(tm>='09:30')&(tm<='16:00')]
            out[s]=x[['datetime','open','high','low','close','volume']].reset_index(drop=True)
        except Exception:out[s]=pd.DataFrame()
    return out


def decision_slot(now):
    m=now.hour*60+now.minute
    if m<720:return '10:00'
    if m<840:return '12:00'
    if m<945:return '14:00'
    return '15:45'


def contexts(symbols,now):
    daily=v4.download_daily(symbols+hist.BENCHMARKS); day={}; week={}; sector={}; market={}
    for b in ['SPY','QQQ','IWM','^VIX','TLT','GLD','USO']:
        d=v4.completed_daily(daily.get(b,pd.DataFrame()),now)
        if not d.empty:
            pref=b.replace('^','').lower(); ctx=v4.context_row(hist.daily_features(d,pref),now)
            for k,v in ctx.items():
                if k.endswith(('ret5','ret20','ret63','atr_pct','rv20')) or 'above_ema20' in k or 'ema20_gt_ema50' in k:market[k]=v
    for s in symbols:
        d=v4.completed_daily(daily.get(s,pd.DataFrame()),now)
        if d.empty:continue
        day[s]=v4.context_row(hist.daily_features(d,'day'),now); week[s]=v4.context_row(hist.weekly_features(d),now)
        sec=hist.SECTOR_OF.get(s); sd=v4.completed_daily(daily.get(sec,pd.DataFrame()),now) if sec else pd.DataFrame(); sc={}
        if sec and not sd.empty:
            allsc=v4.context_row(hist.daily_features(sd,'sector'),now); sc={k:v for k,v in allsc.items() if k in ('sector_ret5','sector_ret20','sector_ret63','sector_above_ema20','sector_ema20_gt_ema50')}
        sector[s]=sc
    return day,week,sector,market


def cross_section(symbols,bars,day,week,spy):
    rows=[]; spyf=hist.intraday_feature_frame(spy) if spy is not None and not spy.empty else pd.DataFrame(); spy_last=spyf.iloc[-1] if not spyf.empty else None
    for s in symbols:
        b=bars.get(s,pd.DataFrame())
        if b.empty or s not in day or s not in week:continue
        f=hist.intraday_feature_frame(b)
        if f.empty:continue
        r=f.iloc[-1]; rec={'symbol':s,'m15_above_ema20':r.get('m15_above_ema20'),'m15_ema20_gt_ema50':r.get('m15_ema20_gt_ema50'),'m15_macd_delta_pct':r.get('m15_macd_delta_pct'),'session_vwap_dist':r.get('session_vwap_dist'),'h4_ret':r.get('h4_ret'),'session_ret':r.get('session_ret'),'session_rvol':r.get('session_rvol'),'m15_ret10':r.get('m15_ret10')}
        rec.update({k:day[s].get(k) for k in ('day_above_ema20','day_ema20_gt_ema50')}); rec.update({k:week[s].get(k) for k in ('week_above_ema20','week_ema20_gt_ema50')}); rows.append(rec)
    if not rows:return {}
    d=pd.DataFrame(rows).set_index('symbol'); d['candidate_score']=hist.score_proxy(d); d['candidate_rank_pct']=d.candidate_score.rank(pct=True); d['activity_rank_pct']=d.session_ret.abs().rank(pct=True); d['rvol_rank_pct']=d.session_rvol.rank(pct=True)
    spyret=float(spy_last.get('m15_ret10',np.nan)) if spy_last is not None else np.nan; d['rel_m15_10_vs_spy']=d.m15_ret10-spyret; d['cross_section_rel_rank']=d.rel_m15_10_vs_spy.rank(pct=True)
    return {s:{k:(float(v) if pd.notna(v) else np.nan) for k,v in row.items() if k in ('candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank')} for s,row in d.iterrows()}


def direct_target(row,h,hp):
    price=float((row.get('quote') or {}).get('price') or 0); up=float(hp.get('p_up') or 0); dn=float(hp.get('p_down') or 0); side='up' if up>=dn else 'down'; stats=(hp.get('historical_return_targets') or {}).get(side)
    if price>0 and stats and int(stats.get('n',0))>=30:
        med=float(stats['median_return']); q20=float(stats['q20_return']); q80=float(stats['q80_return']); lo=min(q20,q80); hi=max(q20,q80)
        return {'horizon_sessions':h,'current_price':round(price,4),'forecast_direction':side.upper(),'direction_probability':round(max(up,dn),4),'central_expected_move_pct':round(med*100,3),'central_target_price':round(price*(1+med),2),'projected_range_low':round(max(.01,price*(1+lo)),2),'projected_range_high':round(price*(1+hi),2),'historical_comparable_target_n':int(stats['n']),'method':'median and 20th/80th percentile realized returns from the frozen independent historical target-calibration block','status':'FROZEN_META_MODEL_TARGET'}
    return prev.build_price_forecast(row,h,hp)


def main():
    prev.main(); payload=json.loads(OUT.read_text()); ready,version=meta_ready()
    if not ready:
        payload['meta_v6_layer']={'status':'BUILDING','truth_note':'Current model remains active until a leakage-controlled intraday meta bundle passes activation audit.'}; OUT.write_text(json.dumps(payload,separators=(',',':'))); return
    symbols=base.UNIVERSE; now=datetime.now(base.NY); broad=bulk_15m(symbols+['SPY']); day,week,sector,market=contexts(symbols,now); ranks=cross_section(symbols,broad,day,week,broad.get('SPY',pd.DataFrame())); slot=decision_slot(now)
    spy=broad.get('SPY',pd.DataFrame()); updated=[]
    for row in payload.get('rows',[]):
        s=row.get('symbol'); b=broad.get(s,pd.DataFrame())
        if row.get('error') or b.empty or spy.empty or s not in day or s not in week:continue
        fv=v4.feature_vector(b,spy,day.get(s,{}),week.get(s,{}),sector.get(s,{}),market); fv.update(ranks.get(s,{})); fv['slot']=slot
        probs=predict(fv); row['probability_model']=probs; row['cross_section_features']=ranks.get(s,{}); row['model_decision_slot']=slot
        forecasts={}
        for hs,hp in (probs.get('horizons') or {}).items():
            h=int(hs); pf=direct_target(row,h,hp)
            if pf: forecasts[hs]=pf; hp['price_forecast']=pf
        row['price_forecasts']=forecasts; gates=[]
        for hs in ('1','5','10'):
            hp=(probs.get('horizons') or {}).get(hs,{})
            for side in ('UP','DOWN'):
                p=hp.get('p_up' if side=='UP' else 'p_down'); th=hp.get('up_threshold' if side=='UP' else 'down_threshold'); passed_key=hp.get('up_threshold_passed' if side=='UP' else 'down_threshold_passed')
                if p is None or th is None:continue
                al,opp,tfs=v4.mtf_alignment(row,side); passed=bool(passed_key and al>=3 and opp<=1 and not row.get('live_stale'))
                gates.append({'horizon_sessions':int(hs),'side':side,'probability':p,'frozen_threshold':th,'mtf_alignment':al,'mtf_opposed':opp,'timeframes':tfs,'comparable_setup_n':(hp.get('comparable_setup_n') or {}).get(side.lower()),'model_dispersion':hp.get('model_dispersion'),'status':'FROZEN_FORWARD_SIGNAL_CANDIDATE' if passed else 'NO_SIGNAL'})
        gates.sort(key=lambda x:(x['status']=='FROZEN_FORWARD_SIGNAL_CANDIDATE',x['probability'],x['mtf_alignment']),reverse=True); row['signal_engine']={'status':gates[0]['status'] if gates else 'NO_SIGNAL','best':gates[0] if gates else None,'all_gates':gates,'policy':'Meta-model threshold + >=60 comparable setups + <=9% model disagreement + 3/4 MTF agreement; final validation remains prospective.'}; updated.append(s)
    valid=[r for r in payload.get('rows',[]) if not r.get('error')]; payload['top_upside']=[r['symbol'] for r in sorted(valid,key=lambda r:((r.get('probability_model',{}).get('horizons',{}).get('5',{}).get('p_up')) or 0),reverse=True)[:3]]; payload['top_downside']=[r['symbol'] for r in sorted(valid,key=lambda r:((r.get('probability_model',{}).get('horizons',{}).get('5',{}).get('p_down')) or 0),reverse=True)[:3]]
    payload['engine']='Phase1 Selective Intraday Meta Forecast Engine V2.1'; payload['status']='META MODEL ACTIVE — FROZEN FOR FORWARD VALIDATION'; payload['ranking_basis']='Calibrated 3-model ensemble, independent comparable-setup regime calibration, cross-sectional rank and abstention.'; payload['meta_v6_layer']={'status':'ACTIVE','model_version':version,'updated_symbols':updated,'broad_intraday_universe':len(broad),'cross_section_features':['candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank'],'price_target_method':'historical conditional return distribution when enough comparable signals exist; ATR fallback otherwise'}
    OUT.write_text(json.dumps(payload,separators=(',',':')))

if __name__=='__main__':main()
