from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import forecast.update_forward_validation as old

LIVE=Path('docs/data/live_forecast.json')
LOG=Path('docs/data/forward_predictions.json')
SUMMARY=Path('docs/data/forward_validation.json')
NY=ZoneInfo('America/New_York')
HORIZONS=(1,5,10)


def append_current(entries, live):
    if not live.get('market',{}).get('market_open'): return entries
    dt=datetime.fromisoformat(live['generated_at'].replace('Z','+00:00')).astimezone(NY)
    slot=old.near_slot(dt)
    if not slot: return entries
    existing={(x.get('date'),x.get('slot'),x.get('symbol'),x.get('model_version')) for x in entries}
    bench=(live.get('benchmark_snapshot') or {}).get('price')
    for r in live.get('rows',[]):
        if r.get('error') or r.get('live_stale') or not (r.get('quote') or {}).get('price'): continue
        pm=r.get('probability_model') or {}; ver=pm.get('model_version'); key=(dt.date().isoformat(),slot,r.get('symbol'),ver)
        if key in existing: continue
        best=(r.get('signal_engine') or {}).get('best') or {}
        rec={
            'date':dt.date().isoformat(),'slot':slot,'generated_at':live.get('generated_at'),'symbol':r.get('symbol'),
            'snapshot_price':float(r['quote']['price']),'benchmark_symbol':'SPY','benchmark_snapshot_price':float(bench) if bench else None,
            'model_version':ver,'model_status':pm.get('status'),'signal_status':(r.get('signal_engine') or {}).get('status'),
            'signal_side':best.get('side'),'signal_horizon':best.get('horizon_sessions'),'horizons':{}
        }
        for h in HORIZONS:
            p=(pm.get('horizons') or {}).get(str(h),{})
            rec['horizons'][str(h)]={
                'p_up':p.get('p_up'),'p_down':p.get('p_down'),'p_neutral':p.get('p_neutral'),
                'probability_threshold_up':p.get('up_threshold'),'probability_threshold_down':p.get('down_threshold'),
                'move_threshold':p.get('target_move_threshold'),'target_definition':p.get('target_definition'),
                'strict_validated':bool(p.get('accepted_for_display')),'outcome_known':False,
                'future_close':None,'benchmark_future_close':None,'stock_return':None,'benchmark_return':None,'excess_return':None,
                'realized_up':None,'realized_down':None,'realized_neutral':None,'brier_up':None,'brier_down':None,'signal_success':None,
            }
        entries.append(rec); existing.add(key)
    return entries


def score_entries(entries):
    symbols=sorted({x.get('symbol') for x in entries if x.get('symbol')}|{'SPY'})
    closes=old.daily_close_history(symbols); spy=closes.get('SPY',pd.Series(dtype=float))
    today=pd.Timestamp(datetime.now(NY).date()); spy_completed=spy[spy.index<today]
    for rec in entries:
        s=rec.get('symbol'); series=closes.get(s,pd.Series(dtype=float));
        if series.empty: continue
        d0=pd.Timestamp(rec['date']); completed=series[series.index<today]; future_dates=completed.index[completed.index>d0]
        v3=str(rec.get('model_version') or '').startswith('intraday-signal-v3')
        for h in HORIZONS:
            obj=(rec.get('horizons') or {}).get(str(h),{})
            if obj.get('outcome_known') or len(future_dates)<h: continue
            fdt=future_dates[h-1]
            if fdt not in completed.index: continue
            fc=float(completed.loc[fdt]); p_up=obj.get('p_up')
            if p_up is None: continue
            if v3:
                sp0=rec.get('benchmark_snapshot_price'); th=obj.get('move_threshold')
                if sp0 is None or th is None or fdt not in spy_completed.index: continue
                sret=fc/float(rec['snapshot_price'])-1; bfc=float(spy_completed.loc[fdt]); bret=bfc/float(sp0)-1; ex=sret-bret
                if abs(sret)>0.55:
                    obj['withheld_reason']='split_like_or_extreme_raw_price_discontinuity'; continue
                yup=1 if ex>float(th) else 0; ydn=1 if ex < -float(th) else 0; yn=1 if not yup and not ydn else 0
                obj.update({'outcome_known':True,'future_date':fdt.date().isoformat(),'future_close':round(fc,6),'benchmark_future_close':round(bfc,6),'stock_return':round(sret,8),'benchmark_return':round(bret,8),'excess_return':round(ex,8),'realized_up':yup,'realized_down':ydn,'realized_neutral':yn,'brier_up':round((float(p_up)-yup)**2,8)})
                if obj.get('p_down') is not None: obj['brier_down']=round((float(obj['p_down'])-ydn)**2,8)
                if rec.get('signal_horizon')==h and rec.get('signal_status')=='FROZEN_FORWARD_SIGNAL_CANDIDATE':
                    side=rec.get('signal_side'); obj['signal_success']=bool((side=='UP' and yup==1) or (side=='DOWN' and ydn==1))
            else:
                realized=1 if fc>float(rec['snapshot_price']) else 0
                obj.update({'outcome_known':True,'future_date':fdt.date().isoformat(),'future_close':round(fc,6),'realized_up':realized,'brier_up':round((float(p_up)-realized)**2,8)})
    return entries


def wilson_lower(successes,n,z=1.96):
    return old.wilson_lower(successes,n,z)


def side_metrics(rows, side):
    pk='p_up' if side=='up' else 'p_down'; yk='realized_up' if side=='up' else 'realized_down'; bk='brier_up' if side=='up' else 'brier_down'
    z=[r for r in rows if r.get(pk) is not None and r.get(yk) is not None]
    if not z: return {'n':0,'status':'COLLECTING'}
    p=np.array([float(r[pk]) for r in z]); y=np.array([int(r[yk]) for r in z]); base=float(y.mean()); bp=np.full(len(y),base)
    bins=[]
    for lo in np.arange(0,1,0.1):
        hi=lo+0.1; m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): bins.append({'lo':round(float(lo),1),'hi':round(float(hi),1),'n':int(m.sum()),'mean_probability':round(float(p[m].mean()),4),'observed_rate':round(float(y[m].mean()),4)})
    return {
        'n':int(len(y)),'base_rate':round(base,4),'mean_probability':round(float(p.mean()),4),'observed_rate':round(float(y.mean()),4),
        'brier':round(float(np.mean((p-y)**2)),6),'base_brier':round(float(np.mean((bp-y)**2)),6),
        'calibration_gap_abs':round(abs(float(p.mean()-y.mean())),4),'calibration_bins':bins,
        'status':'EARLY_FORWARD_SAMPLE' if len(y)<200 else 'FORWARD_SAMPLE_AVAILABLE'
    }


def summarize(entries):
    v3=[e for e in entries if str(e.get('model_version') or '').startswith('intraday-signal-v3')]
    result={
        'generated_at':datetime.now().astimezone().isoformat(),'status':'COLLECTING_FROZEN_V3_FORWARD_EVIDENCE',
        'definition':'For V3, both stock and SPY start at the same logged intraday snapshot. Outcomes are meaningful SPY-relative moves using the frozen volatility-adjusted threshold.',
        'slots_et':[f'{h:02d}:{m:02d}' for h,m in old.SLOTS],'total_logged_snapshots':len(entries),'v3_logged_snapshots':len(v3),'horizons':{}
    }
    for h in HORIZONS:
        matured=[]; sig=[]
        for rec in v3:
            o=(rec.get('horizons') or {}).get(str(h),{})
            if o.get('outcome_known'): matured.append(o)
            if o.get('outcome_known') and o.get('signal_success') is not None: sig.append(bool(o['signal_success']))
        up=side_metrics(matured,'up'); down=side_metrics(matured,'down')
        signal_n=len(sig); signal_precision=float(np.mean(sig)) if sig else None; lower=wilson_lower(sum(sig),signal_n) if sig else None
        result['horizons'][str(h)]={
            'matured_n':len(matured),'up':up,'down':down,
            'frozen_signal_n':signal_n,'frozen_signal_precision':round(signal_precision,4) if signal_precision is not None else None,
            'frozen_signal_wilson_lower_95':round(float(lower),4) if lower is not None else None,
            'status':'COLLECTING' if len(matured)<200 or signal_n<50 else 'FORWARD_EVIDENCE_AVAILABLE'
        }
    return result


def main():
    if not LIVE.exists(): raise RuntimeError('live_forecast.json missing')
    live=old.load_json(LIVE,{}); entries=old.load_json(LOG,[])
    entries=append_current(entries,live); entries=score_entries(entries)
    LOG.parent.mkdir(parents=True,exist_ok=True); LOG.write_text(json.dumps(entries,separators=(',',':'))); SUMMARY.write_text(json.dumps(summarize(entries),separators=(',',':')))
    print(json.dumps({'logged':len(entries),'v3_logged':sum(str(e.get('model_version') or '').startswith('intraday-signal-v3') for e in entries),'summary':str(SUMMARY)},indent=2))

if __name__=='__main__': main()
