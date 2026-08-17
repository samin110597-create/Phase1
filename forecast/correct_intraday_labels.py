from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('forecast/data/intraday_live_like_v2.csv.gz')
CACHE = Path('forecast/cache/intraday15')
SUMMARY = Path('docs/data/intraday_training_summary.json')
SLOTS = ['10:00','12:00','14:00','15:45']


def load_bars(symbol: str):
    p=CACHE/f'{symbol.replace("^","IDX_")}.csv.gz'
    if not p.exists(): return pd.DataFrame()
    x=pd.read_csv(p,compression='gzip',parse_dates=['datetime'])
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['datetime','open','close']).sort_values('datetime')


def session_closes(bars: pd.DataFrame):
    x=bars.copy(); x['date']=x['datetime'].dt.normalize()
    return x.groupby('date').tail(1).set_index('date')['close'].sort_index()


def decision_prices(bars: pd.DataFrame):
    """Price proxy observable at the exact decision timestamp.

    Prefer the opening price of the 15-minute bar that begins at the decision
    time. If that bar is missing, fall back to the last fully completed bar's
    close. Indicators remain based only on completed bars; this function is
    used only to align the outcome's starting price with the live logger.
    """
    x=bars.copy(); x['date']=x['datetime'].dt.normalize(); rows=[]
    for date,g in x.groupby('date',sort=True):
        g=g.sort_values('datetime')
        for slot in SLOTS:
            decision=pd.Timestamp(f'{date.date()} {slot}')
            exact=g[g['datetime']==decision]
            if not exact.empty and pd.notna(exact.iloc[0]['open']):
                rows.append((decision,float(exact.iloc[0]['open'])))
                continue
            cutoff=decision-pd.Timedelta(minutes=15)
            e=g[g['datetime']<=cutoff]
            if not e.empty:
                rows.append((decision,float(e.iloc[-1]['close'])))
    return pd.Series({dt:px for dt,px in rows},dtype=float)


def main():
    if not DATA.exists(): raise RuntimeError('dataset missing')
    df=pd.read_csv(DATA,compression='gzip',parse_dates=['snapshot_dt'])
    spy=load_bars('SPY')
    if spy.empty: raise RuntimeError('SPY cache missing')
    spy_close=session_closes(spy); spy_snap=decision_prices(spy)
    changed=0; missing=0
    for symbol,idx in df.groupby('symbol').groups.items():
        bars=load_bars(symbol)
        if bars.empty:
            missing += len(idx); continue
        sc=session_closes(bars); stock_snap=decision_prices(bars); dates=list(sc.index); pos={d:i for i,d in enumerate(dates)}
        for i in idx:
            dt=pd.Timestamp(df.at[i,'snapshot_dt']); d=dt.normalize(); k=pos.get(d)
            sp0=spy_snap.get(dt,np.nan); stock0=stock_snap.get(dt,np.nan)
            # Preserve the feature close separately; the target starts from the
            # decision-time price proxy, which is closer to the live quote used
            # by the prospective logger.
            if pd.notna(stock0):
                df.at[i,'decision_price_proxy']=float(stock0)
            for h in (1,5,10):
                if k is None or k+h>=len(dates) or pd.isna(sp0) or pd.isna(stock0):
                    for c in (f'fwd_{h}_return',f'spy_fwd_{h}_return',f'fwd_{h}_excess',f'label_up_{h}',f'label_down_{h}'):
                        df.at[i,c]=np.nan
                    continue
                fd=dates[k+h]
                if fd not in spy_close.index:
                    continue
                sr=float(sc.loc[fd]/float(stock0)-1); pr=float(spy_close.loc[fd]/float(sp0)-1); ex=sr-pr
                atr=float(df.at[i,'day_atr_pct']) if 'day_atr_pct' in df and pd.notna(df.at[i,'day_atr_pct']) else 0.02
                th=max(0.003,0.35*max(0.005,atr)*math.sqrt(h)); bad=abs(sr)>0.55
                df.at[i,f'fwd_{h}_return']=sr; df.at[i,f'spy_fwd_{h}_return']=pr; df.at[i,f'fwd_{h}_excess']=ex; df.at[i,f'move_threshold_{h}']=th
                df.at[i,f'label_up_{h}']=np.nan if bad else float(ex>th); df.at[i,f'label_down_{h}']=np.nan if bad else float(ex < -th)
                changed += 1
    df.to_csv(DATA,index=False,compression='gzip')
    if SUMMARY.exists():
        s=json.loads(SUMMARY.read_text())
        s['benchmark_label_alignment']='CORRECTED: stock and SPY forward returns both begin at the same observable decision-time price proxy (decision bar open when available); indicators still use only fully completed bars; future session close is the common endpoint.'
        s['label_correction_values_written']=changed
        s['label_correction_missing_rows']=missing
        SUMMARY.write_text(json.dumps(s,separators=(',',':')))
    print(json.dumps({'status':'labels aligned to observable decision-time stock and SPY prices','values_written':changed,'missing_rows':missing},indent=2))

if __name__=='__main__': main()
