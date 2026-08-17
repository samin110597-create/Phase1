from __future__ import annotations

import json, sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v4 as v4
import scripts.github_live_forecast_v8 as prev
import forecast.build_intraday_live_like_dataset as hist
import forecast.train_hourly_meta_model_v7 as hv7
from forecast.live_intraday_v3_inference import predict

OUT=Path('docs/data/live_forecast.json')
MODEL=Path('forecast/data/intraday_signal_v3.joblib')


def hourly_ready():
    try:
        return MODEL.exists() and joblib.load(MODEL).get('version')=='hourly-meta-v7'
    except Exception:return False


def aggregate_1h(bars):
    if bars is None or bars.empty:return pd.DataFrame()
    z=bars.copy().sort_values('datetime'); z['datetime']=pd.to_datetime(z.datetime,errors='coerce'); z=z.dropna(subset=['datetime']); z['date']=z.datetime.dt.normalize(); z['bar_no']=z.groupby('date').cumcount(); z['bucket']=(z.bar_no//4).astype(int)
    out=z.groupby(['date','bucket'],as_index=False).agg(datetime=('datetime','first'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
    # Require a full four-bar hour for regular-session model parity.
    counts=z.groupby(['date','bucket']).size().rename('n').reset_index(); out=out.merge(counts,on=['date','bucket']); out=out[out.n>=4]
    return out[['datetime','open','high','low','close','volume']].sort_values('datetime').reset_index(drop=True)


def current_slot(now):
    m=now.hour*60+now.minute
    if m<630:return None
    if m<750:return '10:30'
    if m<870:return '12:30'
    if m<930:return '14:30'
    return '15:30'


def hour_vector(stock,spy,day_ctx,week_ctx,sector_ctx,market_ctx,rank_ctx,slot):
    z=hv7.hourly_features(stock); s=hv7.hourly_features(spy)
    if z.empty or s.empty:return {}
    r=z.iloc[-1]; sr=s.iloc[-1]; keys=['h1_ret1','h1_ret3','h1_ret6','h1_rsi14','h1_above_ema20','h1_ema20_gt_ema50','h1_macd_delta_pct','h1_atr_pct','h1_vol_ratio20','h1_vwap20_dist','h4_ret','h4_ret3','h4_range_pct','h4_vol_ratio','session_ret','session_range_pos','session_vwap_dist','session_rvol','volume_accel']
    f={k:(float(r[k]) if k in r and pd.notna(r[k]) else np.nan) for k in keys}; f.update(day_ctx); f.update(week_ctx); f.update(sector_ctx); f.update(market_ctx)
    f['spy_h1_ret3']=float(sr.get('h1_ret3',np.nan)); f['spy_h1_ret6']=float(sr.get('h1_ret6',np.nan)); f['spy_h4_ret']=float(sr.get('h4_ret',np.nan)); f['spy_session_ret']=float(sr.get('session_ret',np.nan)); f['spy_session_vwap_dist']=float(sr.get('session_vwap_dist',np.nan)); f['spy_session_rvol']=float(sr.get('session_rvol',np.nan)); f['rel_h1_6_vs_spy']=f['h1_ret6']-f['spy_h1_ret6']; f['rel_h4_vs_spy']=f['h4_ret']-f['spy_h4_ret']
    if 'day_ret20' in f and 'sector_ret20' in f and pd.notna(f['day_ret20']) and pd.notna(f['sector_ret20']):f['rel20_vs_sector']=f['day_ret20']-f['sector_ret20']
    f.update(rank_ctx); f['slot']=slot; return f


def hourly_ranks(symbols,hourly,day,week,spy):
    rows=[]; sf=hv7.hourly_features(spy) if spy is not None and not spy.empty else pd.DataFrame(); sr=sf.iloc[-1] if not sf.empty else None
    for s in symbols:
        b=hourly.get(s,pd.DataFrame())
        if b.empty or s not in day or s not in week:continue
        f=hv7.hourly_features(b)
        if f.empty:continue
        r=f.iloc[-1]; rec={'symbol':s,'h1_above_ema20':r.get('h1_above_ema20'),'h1_ema20_gt_ema50':r.get('h1_ema20_gt_ema50'),'h1_macd_delta_pct':r.get('h1_macd_delta_pct'),'session_vwap_dist':r.get('session_vwap_dist'),'h4_ret':r.get('h4_ret'),'session_ret':r.get('session_ret'),'session_rvol':r.get('session_rvol'),'h1_ret6':r.get('h1_ret6'),'day_above_ema20':day[s].get('day_above_ema20'),'day_ema20_gt_ema50':day[s].get('day_ema20_gt_ema50'),'week_above_ema20':week[s].get('week_above_ema20'),'week_ema20_gt_ema50':week[s].get('week_ema20_gt_ema50')}; rows.append(rec)
    if not rows:return {}
    d=pd.DataFrame(rows).set_index('symbol'); d['candidate_score']=hv7.candidate_score(d); d['candidate_rank_pct']=d.candidate_score.rank(pct=True); d['activity_rank_pct']=d.session_ret.abs().rank(pct=True); d['rvol_rank_pct']=d.session_rvol.rank(pct=True); spyret=float(sr.get('h1_ret6',np.nan)) if sr is not None else np.nan; d['rel_h1_6_vs_spy']=d.h1_ret6-spyret; d['cross_section_rel_rank']=d.rel_h1_6_vs_spy.rank(pct=True)
    return {s:{k:(float(v) if pd.notna(v) else np.nan) for k,v in row.items() if k in ('candidate_score','candidate_rank_pct','activity_rank_pct','rvol_rank_pct','cross_section_rel_rank')} for s,row in d.iterrows()}


def main():
    prev.main(); payload=json.loads(OUT.read_text())
    if not hourly_ready():
        payload['hourly_meta_v7_layer']={'status':'BUILDING','truth_note':'Old probability model remains visible only until the bulk-hourly V7 bundle is trained and passes its audit.'}; OUT.write_text(json.dumps(payload,separators=(',',':'))); return
    now=datetime.now(base.NY); slot=current_slot(now)
    if slot is None and payload.get('market',{}).get('market_open'):
        payload['hourly_meta_v7_layer']={'status':'WAITING_FOR_FIRST_COMPLETE_HOUR','truth_note':'The V7 model only evaluates after a complete regular-session hour exists.'}; OUT.write_text(json.dumps(payload,separators=(',',':'))); return
    symbols=base.UNIVERSE; bars15=prev.bulk_15m(symbols+['SPY']); hourly={s:aggregate_1h(b) for s,b in bars15.items()}; day,week,sector,market=prev.contexts(symbols,now); spy=hourly.get('SPY',pd.DataFrame()); ranks=hourly_ranks(symbols,hourly,day,week,spy); updated=[]
    for row in payload.get('rows',[]):
        s=row.get('symbol'); hb=hourly.get(s,pd.DataFrame())
        if row.get('error') or hb.empty or spy.empty or s not in day or s not in week:continue
        fv=hour_vector(hb,spy,day.get(s,{}),week.get(s,{}),sector.get(s,{}),market,ranks.get(s,{}),slot or '15:30'); probs=predict(fv); row['probability_model']=probs; row['cross_section_features']=ranks.get(s,{}); row['model_decision_slot']=slot or '15:30'; forecasts={}
        for hs,hp in (probs.get('horizons') or {}).items():
            pf=prev.direct_target(row,int(hs),hp)
            if pf:forecasts[hs]=pf; hp['price_forecast']=pf
        row['price_forecasts']=forecasts; gates=[]
        for hs in ('1','5','10'):
            hp=(probs.get('horizons') or {}).get(hs,{})
            for side in ('UP','DOWN'):
                p=hp.get('p_up' if side=='UP' else 'p_down'); th=hp.get('up_threshold' if side=='UP' else 'down_threshold'); pass0=hp.get('up_threshold_passed' if side=='UP' else 'down_threshold_passed')
                if p is None or th is None:continue
                al,opp,tfs=v4.mtf_alignment(row,side); passed=bool(pass0 and al>=3 and opp<=1 and not row.get('live_stale'))
                gates.append({'horizon_sessions':int(hs),'side':side,'probability':p,'frozen_threshold':th,'mtf_alignment':al,'mtf_opposed':opp,'timeframes':tfs,'comparable_setup_n':(hp.get('comparable_setup_n') or {}).get(side.lower()),'model_dispersion':hp.get('model_dispersion'),'cross_section_rank':(ranks.get(s,{}) or {}).get('candidate_rank_pct'),'status':'FROZEN_FORWARD_SIGNAL_CANDIDATE' if passed else 'NO_SIGNAL'})
        gates.sort(key=lambda x:(x['status']=='FROZEN_FORWARD_SIGNAL_CANDIDATE',x['probability'],x['mtf_alignment']),reverse=True); row['signal_engine']={'status':gates[0]['status'] if gates else 'NO_SIGNAL','best':gates[0] if gates else None,'all_gates':gates,'policy':'Hourly-meta V7 threshold + comparable-history count + ensemble agreement + cross-sectional rank + live 15m/4h/Day/Week confirmation.'}; updated.append(s)
    valid=[r for r in payload.get('rows',[]) if not r.get('error')]; payload['top_upside']=[r['symbol'] for r in sorted(valid,key=lambda r:((r.get('probability_model',{}).get('horizons',{}).get('5',{}).get('p_up')) or 0),reverse=True)[:3]]; payload['top_downside']=[r['symbol'] for r in sorted(valid,key=lambda r:((r.get('probability_model',{}).get('horizons',{}).get('5',{}).get('p_down')) or 0),reverse=True)[:3]]; payload['engine']='Phase1 Selective Hourly/15m Meta Forecast Engine V2.2'; payload['status']='HOURLY META V7 ACTIVE — FROZEN FOR PROSPECTIVE VALIDATION'; payload['ranking_basis']='Recent bulk-hourly 3-class ensemble + independent comparable-setup calibration + 40-stock cross-sectional rank + live 15m confirmation + abstention.'; payload['hourly_meta_v7_layer']={'status':'ACTIVE','model_version':'hourly-meta-v7','updated_symbols':updated,'hourly_universe_rows':len([x for x in hourly.values() if not x.empty]),'decision_slot':slot,'price_target_method':'independent historical conditional return distribution when available; volatility fallback otherwise'}
    OUT.write_text(json.dumps(payload,separators=(',',':')))

if __name__=='__main__':main()
