from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v3 as legacy
import forecast.build_intraday_live_like_dataset as hist
from forecast.live_intraday_v3_inference import available as v3_available, predict as v3_predict

OUT = Path('docs/data/live_forecast.json')

_original_quote = base.finnhub_quote

def resilient_quote(symbol: str):
    last=None
    for delay in (0.0,0.8,2.0):
        if delay: time.sleep(delay)
        try: return _original_quote(symbol)
        except Exception as e: last=e
    raise last if last else RuntimeError('quote failed')

base.finnhub_quote = resilient_quote


def recent_15m(symbol: str, outputsize=220):
    if not base.TWELVE_KEY: raise RuntimeError('TWELVE_DATA_API_KEY missing')
    u='https://api.twelvedata.com/time_series?'+urlencode({
        'symbol':symbol,'interval':'15min','outputsize':outputsize,'timezone':'America/New_York','adjust':'splits','apikey':base.TWELVE_KEY
    })
    d=base.get_json(u); vals=d.get('values') if isinstance(d,dict) else None
    if not isinstance(vals,list) or not vals: raise RuntimeError((d.get('message') if isinstance(d,dict) else None) or 'no 15m bars')
    x=pd.DataFrame(vals); x['datetime']=pd.to_datetime(x['datetime'],errors='coerce')
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x.get(c),errors='coerce')
    return x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime')[['datetime','open','high','low','close','volume']].reset_index(drop=True)


def download_daily(symbols):
    raw=yf.download(sorted(set(symbols)),period='18mo',auto_adjust=True,actions=False,group_by='ticker',threads=True,progress=False)
    out={}
    for s in sorted(set(symbols)):
        try:
            x=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy()
            x=x.rename(columns=str.lower).dropna(subset=['close']); x.index=pd.to_datetime(x.index).tz_localize(None)
            out[s]=x[['open','high','low','close','volume']].astype(float)
        except Exception: out[s]=pd.DataFrame()
    return out


def completed_daily(d: pd.DataFrame, now_ny: datetime):
    if d is None or d.empty: return d
    today=pd.Timestamp(now_ny.date()); minute=now_ny.hour*60+now_ny.minute
    # During premarket/regular session, today's daily candle is incomplete and must not enter Day/Week features.
    use_today=now_ny.weekday()<5 and minute>=960
    return d[d.index <= today] if use_today else d[d.index < today]


def context_row(f: pd.DataFrame, now_ny: datetime, weekly=False):
    if f is None or f.empty: return {}
    today=pd.Timestamp(now_ny.date()); minute=now_ny.hour*60+now_ny.minute
    use_today=now_ny.weekday()<5 and minute>=960
    z=f[f.index <= today] if use_today else f[f.index < today]
    if z.empty: return {}
    r=z.iloc[-1]
    return {k:(float(v) if pd.notna(v) else np.nan) for k,v in r.items()}


def completed_intraday(bars: pd.DataFrame, now_ny: datetime, market_open: bool):
    if bars is None or bars.empty or not market_open: return bars
    minute=now_ny.hour*60+now_ny.minute; floor=(minute//15)*15; completed_start=floor-15
    if completed_start < 570: return bars[bars['datetime'].dt.normalize() < pd.Timestamp(now_ny.date())]
    cutoff=pd.Timestamp(now_ny.date())+pd.Timedelta(minutes=completed_start)
    z=bars[bars['datetime']<=cutoff]
    return z if not z.empty else bars


def display_day(d: pd.DataFrame, label: str):
    if d is None or d.empty: return None
    x=d.reset_index().rename(columns={d.index.name or 'index':'datetime'})
    return base.frame_features(x,label)


def feature_vector(stock_bars, spy_bars, day_ctx, week_ctx, sector_ctx, market_ctx):
    z=hist.intraday_feature_frame(stock_bars); s=hist.intraday_feature_frame(spy_bars)
    if z.empty or s.empty: return {}
    r=z.iloc[-1]; sr=s.iloc[-1]; keys=[
        'm15_ret1','m15_ret3','m15_ret10','m15_rsi14','m15_above_ema20','m15_ema20_gt_ema50','m15_macd_delta_pct','m15_atr_pct','m15_vol_ratio20','m15_vwap20_dist',
        'h4_ret','h4_ret3','h4_range_pct','h4_vol_ratio','session_ret','session_range_pos','session_vwap_dist','session_rvol','volume_accel','opening_range_pos'
    ]
    f={k:(float(r[k]) if k in r and pd.notna(r[k]) else np.nan) for k in keys}
    f.update(day_ctx); f.update(week_ctx); f.update(sector_ctx); f.update(market_ctx)
    f['spy_m15_ret3']=float(sr.get('m15_ret3',np.nan)); f['spy_m15_ret10']=float(sr.get('m15_ret10',np.nan)); f['spy_h4_ret']=float(sr.get('h4_ret',np.nan))
    f['spy_session_ret']=float(sr.get('session_ret',np.nan)); f['spy_session_vwap_dist']=float(sr.get('session_vwap_dist',np.nan)); f['spy_session_rvol']=float(sr.get('session_rvol',np.nan))
    f['rel_m15_10_vs_spy']=f['m15_ret10']-f['spy_m15_ret10']; f['rel_h4_vs_spy']=f['h4_ret']-f['spy_h4_ret']
    if 'day_ret20' in f and 'sector_ret20' in f and pd.notna(f['day_ret20']) and pd.notna(f['sector_ret20']): f['rel20_vs_sector']=f['day_ret20']-f['sector_ret20']
    return f


def mtf_alignment(row, side):
    vals=[]; m=row.get('m15') or {}; h=row.get('h4') or {}; d=row.get('day') or {}; w=row.get('week') or {}
    if m:
        vals.append('UP' if m.get('above_ema20') and (m.get('macd_delta_pct') or 0)>=0 else 'DOWN' if (not m.get('above_ema20')) and (m.get('macd_delta_pct') or 0)<0 else 'NEUTRAL')
    if h and h.get('return_4h_pct') is not None: vals.append('UP' if h['return_4h_pct']>0 else 'DOWN' if h['return_4h_pct']<0 else 'NEUTRAL')
    if d: vals.append('UP' if d.get('above_ema20') and d.get('ema20_above_ema50') else 'DOWN' if (not d.get('above_ema20')) and (not d.get('ema20_above_ema50')) else 'NEUTRAL')
    if w: vals.append('UP' if w.get('above_ema20') and w.get('ema20_above_ema50') else 'DOWN' if (not w.get('above_ema20')) and (not w.get('ema20_above_ema50')) else 'NEUTRAL')
    return sum(v==side for v in vals), sum(v not in (side,'NEUTRAL') for v in vals), vals


def fallback_with_upgrade_status():
    legacy.engine.main()
    p=json.loads(OUT.read_text()); p['v3_upgrade_status']='HISTORICAL LIVE-LIKE MODEL BUILDING'; p['engine']='Phase1 GitHub-Only Live Forecast Engine V2 + V3 BUILDING'
    p['truth_policy']=p.get('truth_policy',{}); p['truth_policy']['v3_not_used_until_model_file_exists']=True
    OUT.write_text(json.dumps(p,separators=(',',':')))


def main():
    if not v3_available():
        fallback_with_upgrade_status(); return
    if not base.FINNHUB_KEY or not base.TWELVE_KEY: raise RuntimeError('GitHub market-data secrets missing')
    state=base.market_state(); now_ny=datetime.now(base.NY); market_open=bool(state['market_open']); minute=now_ny.hour*60+now_ny.minute
    symbols=base.UNIVERSE; daily=download_daily(symbols+hist.BENCHMARKS)

    market_ctx={}
    for b in ['SPY','QQQ','IWM','^VIX','TLT','GLD','USO']:
        d=completed_daily(daily.get(b,pd.DataFrame()),now_ny)
        if not d.empty:
            pref=b.replace('^','').lower(); f=hist.daily_features(d,pref); ctx=context_row(f,now_ny)
            for k,v in ctx.items():
                if k.endswith(('ret5','ret20','ret63','atr_pct','rv20')) or 'above_ema20' in k or 'ema20_gt_ema50' in k: market_ctx[k]=v

    quote_rows={}; quote_errors={}
    for s in symbols:
        try: quote_rows[s]=resilient_quote(s)
        except Exception as e: quote_errors[s]=type(e).__name__
        time.sleep(0.05)

    day_cache={}; week_cache={}; sector_cache={}; pre=[]
    spy_day_ctx={}
    spy_d=completed_daily(daily.get('SPY',pd.DataFrame()),now_ny)
    if not spy_d.empty: spy_day_ctx=context_row(hist.daily_features(spy_d,'day'),now_ny)
    for s in symbols:
        d=completed_daily(daily.get(s,pd.DataFrame()),now_ny); q=quote_rows.get(s)
        if d.empty or not q: continue
        df=hist.daily_features(d,'day'); wf=hist.weekly_features(d); dc=context_row(df,now_ny); wc=context_row(wf,now_ny)
        if not dc or not wc: continue
        day_cache[s]=dc; week_cache[s]=wc
        sec=hist.SECTOR_OF.get(s); sd=completed_daily(daily.get(sec,pd.DataFrame()),now_ny) if sec else pd.DataFrame()
        sc={}
        if sec and not sd.empty:
            sf=hist.daily_features(sd,'sector'); allsc=context_row(sf,now_ny); sc={k:v for k,v in allsc.items() if k in ('sector_ret5','sector_ret20','sector_ret63','sector_above_ema20','sector_ema20_gt_ema50')}
        sector_cache[s]=sc
        score=0.0
        score += 10 if dc.get('day_above_ema20',0)>0.5 else -10; score += 10 if dc.get('day_ema20_gt_ema50',0)>0.5 else -10
        rv=dc.get('day_rsi14'); score += 6 if pd.notna(rv) and rv>=55 else -6 if pd.notna(rv) and rv<=45 else 0
        md=dc.get('day_macd_delta_pct'); score += 7 if pd.notna(md) and md>=0 else -7
        if pd.notna(dc.get('day_ret20')) and pd.notna(spy_day_ctx.get('day_ret20')): score += 8 if dc['day_ret20']>=spy_day_ctx['day_ret20'] else -8
        score += 7 if wc.get('week_above_ema20',0)>0.5 else -7; score += 6 if wc.get('week_ema20_gt_ema50',0)>0.5 else -6
        if q.get('previous_close'): score += max(-10,min(10,(q['price']/q['previous_close']-1)*100*3))
        adv=float((d['close']*d['volume']).tail(20).mean()) if len(d)>=20 else 0
        if q['price']>=5 and adv>=100_000_000: pre.append({'symbol':s,'score':score,'activity':abs(q['price']/q['previous_close']-1) if q.get('previous_close') else 0,'adv20':adv})

    bull=sorted(pre,key=lambda x:x['score'],reverse=True)[:3]; bear=sorted(pre,key=lambda x:x['score'])[:3]; used={x['symbol'] for x in bull+bear}
    wildcard=next((x for x in sorted(pre,key=lambda x:x['activity'],reverse=True) if x['symbol'] not in used),None)
    shortlist=list(dict.fromkeys(x['symbol'] for x in bull+bear+([wildcard] if wildcard else [])))[:7]

    intraday={}; intraday_errors={}
    for s in shortlist+['SPY']:
        try: intraday[s]=completed_intraday(recent_15m(s),now_ny,market_open)
        except Exception as e: intraday_errors[s]=str(e)[:140]
        time.sleep(0.15)
    if 'SPY' not in intraday or intraday['SPY'].empty: raise RuntimeError('SPY 15m context unavailable')
    spy15_display=base.frame_features(intraday['SPY'],'15m')

    rows=[]
    for s in shortlist:
        if s not in intraday or intraday[s].empty:
            rows.append({'symbol':s,'error':intraday_errors.get(s,'no intraday data')}); continue
        d=completed_daily(daily.get(s,pd.DataFrame()),now_ny); raw_day=d.reset_index().rename(columns={d.index.name or 'index':'datetime'}) if not d.empty else pd.DataFrame()
        m15=base.frame_features(intraday[s],'15m'); h4=base.rolling_4h(intraday[s]); day=base.frame_features(raw_day,'day') if not raw_day.empty else None
        wraw=base.weekly_from_daily(raw_day) if not raw_day.empty else pd.DataFrame(); week=base.frame_features(wraw,'week') if not wraw.empty else None
        q=quote_rows.get(s); edge,upscore,downscore,bias,reasons=base.final_score(day,week,m15,h4,q,spy15_display)
        fv=feature_vector(intraday[s],intraday['SPY'],day_cache.get(s,{}),week_cache.get(s,{}),sector_cache.get(s,{}),market_ctx)
        probs=v3_predict(fv)
        stale=bool(market_open and q and q.get('age_seconds') is not None and q['age_seconds']>180)
        row={'symbol':s,'quote':q,'m15':m15,'h4':h4,'day':day,'week':week,'signed_edge':edge,'upside_score':upscore,'downside_score':downscore,'bias':bias,'reasons':reasons,'live_stale':stale,'forecast_status':'MARKET_CLOSED' if not market_open else 'STALE_QUOTE' if stale else 'V3_NEAR_LIVE','probability_model':probs}
        gates=[]
        for h in ('5','10'):
            hp=probs.get('horizons',{}).get(h,{})
            for side in ('UP','DOWN'):
                key='p_up' if side=='UP' else 'p_down'; tk='up_threshold' if side=='UP' else 'down_threshold'; p=hp.get(key); th=hp.get(tk)
                if p is None or th is None: continue
                aligned,opposed,tfs=mtf_alignment(row,side)
                passed=bool(p>=th and not stale and aligned>=3 and opposed<=1 and (not market_open or minute>=600))
                gates.append({'horizon_sessions':int(h),'side':side,'probability':p,'frozen_threshold':th,'mtf_alignment':aligned,'mtf_opposed':opposed,'timeframes':tfs,'status':'FROZEN_FORWARD_SIGNAL_CANDIDATE' if passed else 'NO_SIGNAL'})
        gates.sort(key=lambda x:(x['status']=='FROZEN_FORWARD_SIGNAL_CANDIDATE',x['probability'],x['mtf_alignment']),reverse=True)
        row['signal_engine']={'status':gates[0]['status'] if gates else 'NO_SIGNAL','best':gates[0] if gates else None,'all_gates':gates,'policy':'V3 candidate only when frozen probability threshold and at least 3/4 MTF directions agree; prospective validation is still required.'}
        rows.append(row)

    valid=[r for r in rows if not r.get('error') and not r.get('live_stale')]
    def p(r,h,side): return (r.get('probability_model') or {}).get('horizons',{}).get(str(h),{}).get('p_'+side)
    up5=[(p(r,5,'up'),r) for r in valid if p(r,5,'up') is not None]; dn5=[(p(r,5,'down'),r) for r in valid if p(r,5,'down') is not None]
    top_up=[r['symbol'] for _,r in sorted(up5,key=lambda x:x[0],reverse=True)[:3]]; top_dn=[r['symbol'] for _,r in sorted(dn5,key=lambda x:x[0],reverse=True)[:3]]
    sig=[r for r in valid if (r.get('signal_engine') or {}).get('status')=='FROZEN_FORWARD_SIGNAL_CANDIDATE']
    sig.sort(key=lambda r:((r['signal_engine']['best'] or {}).get('probability') or 0),reverse=True)
    payload={
        'generated_at':base.now_utc().isoformat(),'engine':'Phase1 GitHub-Only Live Forecast Engine V3','mode':'GitHub Actions near-live; V3 live-like model uses latest fully completed 15m bar','status':'V3 FROZEN FOR PROSPECTIVE VALIDATION — SIGNAL CANDIDATES ARE NOT YET FORWARD-VALIDATED',
        'market':state,'sources':{'live_quote':'Finnhub','intraday':'Twelve Data 15m','day_week_regime':'Yahoo Finance adjusted daily','benchmark_intraday':'SPY','market_context':'SPY/QQQ/IWM/VIX/TLT/GLD/USO + sector ETFs'},
        'universe_size':len(symbols),'funnel':{'quotes_received':len(quote_rows),'tradeable_after_liquidity_filter':len(pre),'intraday_shortlist':shortlist,'displayed_max_each_direction':3},
        'freshness_rule':'During regular hours, quote age >180 seconds blocks a signal candidate. V3 uses only fully completed 15-minute bars.',
        'research_top_upside':top_up,'research_top_downside':top_dn,'top_upside':top_up,'top_downside':top_dn,
        'qualified_signal_candidates':[r['symbol'] for r in sig[:3]],'signal_watchlist':[],'signal_state':'FROZEN FORWARD SIGNAL CANDIDATE' if sig else 'NO QUALIFIED SIGNAL','accepted_probability_horizons':[],
        'ranking_basis':'V3 calibrated research probabilities from live-like historical intraday snapshots; final validity must come from frozen prospective outcomes.',
        'rows':rows,'errors':{'quote_count':len(quote_errors),'quote':quote_errors,'intraday':intraday_errors},
        'truth_policy':{'v3_live_training_feature_match':True,'uses_only_completed_15m_bars':True,'day_week_lookahead_blocked':True,'research_probabilities_visible':True,'prospective_validation_required':True,'signal_engine_can_abstain':True}
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({'status':payload['status'],'quotes_received':len(quote_rows),'shortlist':shortlist,'top_upside':top_up,'top_downside':top_dn,'signal_state':payload['signal_state'],'signal_candidates':payload['qualified_signal_candidates']},indent=2))

if __name__=='__main__': main()
