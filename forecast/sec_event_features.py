from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SEC_HEADERS={
    'User-Agent':'Phase1ForecastResearch/1.0 github.com/samin110597-create/Phase1',
    'Accept-Encoding':'gzip, deflate',
}
MAP_URL='https://www.sec.gov/files/company_tickers.json'
SUB_URL='https://data.sec.gov/submissions/CIK{cik:010d}.json'
SUB_FILE_URL='https://data.sec.gov/submissions/{name}'
CACHE=Path('forecast/cache/sec_submissions')
CACHE.mkdir(parents=True,exist_ok=True)

MAJOR_FORMS={'8-K','8-K/A','10-Q','10-Q/A','10-K','10-K/A','6-K','6-K/A','20-F','20-F/A','40-F','40-F/A'}


def _get_json(url:str,tries:int=4):
    last=None
    for attempt in range(tries):
        try:
            r=requests.get(url,headers=SEC_HEADERS,timeout=30)
            if r.status_code in (429,403,503):
                time.sleep(1.2*(attempt+1)); last=RuntimeError(f'SEC HTTP {r.status_code}'); continue
            r.raise_for_status(); time.sleep(.12); return r.json()
        except Exception as e:
            last=e; time.sleep(.8*(attempt+1))
    raise last or RuntimeError('SEC request failed')


def ticker_cik_map()->dict[str,int]:
    p=CACHE/'company_tickers.json'
    if p.exists():
        try: raw=json.loads(p.read_text())
        except Exception: raw=None
    else: raw=None
    if raw is None:
        raw=_get_json(MAP_URL); p.write_text(json.dumps(raw,separators=(',',':')))
    out={}
    values=raw.values() if isinstance(raw,dict) else raw
    for x in values:
        try: out[str(x['ticker']).upper()]=int(x['cik_str'])
        except Exception: pass
    return out


def _recent_frame(obj:dict)->pd.DataFrame:
    rec=((obj.get('filings') or {}).get('recent') or {})
    if not rec:return pd.DataFrame(columns=['form','filingDate','acceptanceDateTime','accessionNumber'])
    n=max((len(v) for v in rec.values() if isinstance(v,list)),default=0); rows=[]
    for i in range(n):
        row={}
        for k,v in rec.items():
            if isinstance(v,list): row[k]=v[i] if i<len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _archive_frame(obj:dict)->pd.DataFrame:
    if not obj:return pd.DataFrame()
    n=max((len(v) for v in obj.values() if isinstance(v,list)),default=0); rows=[]
    for i in range(n):
        row={}
        for k,v in obj.items():
            if isinstance(v,list): row[k]=v[i] if i<len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


def filings_for_symbol(symbol:str,refresh:bool=False)->pd.DataFrame:
    symbol=symbol.upper(); mp=ticker_cik_map(); cik=mp.get(symbol)
    if cik is None:return pd.DataFrame(columns=['event_dt','form'])
    p=CACHE/f'{symbol}_{cik}.json'
    if p.exists() and not refresh:
        try:
            saved=json.loads(p.read_text()); return pd.DataFrame(saved).assign(event_dt=lambda x:pd.to_datetime(x.event_dt,errors='coerce'))
        except Exception: pass
    obj=_get_json(SUB_URL.format(cik=cik)); frames=[_recent_frame(obj)]
    for f in ((obj.get('filings') or {}).get('files') or []):
        name=f.get('name')
        if not name:continue
        try: frames.append(_archive_frame(_get_json(SUB_FILE_URL.format(name=name))))
        except Exception: continue
    x=pd.concat([z for z in frames if not z.empty],ignore_index=True) if any(not z.empty for z in frames) else pd.DataFrame()
    if x.empty:return pd.DataFrame(columns=['event_dt','form'])
    x['form']=x.get('form','').astype(str).str.upper()
    x=x[x.form.isin(MAJOR_FORMS)].copy()
    acc=pd.to_datetime(x.get('acceptanceDateTime'),errors='coerce',utc=True)
    filing=pd.to_datetime(x.get('filingDate'),errors='coerce')
    # If exact acceptance time is unavailable, defer the filing to the next session
    # rather than treating it as known at the morning decision time.
    fallback=(filing+pd.Timedelta(days=1)).dt.tz_localize('America/New_York',nonexistent='shift_forward',ambiguous='NaT').dt.tz_convert('UTC')
    event=acc.fillna(fallback)
    out=pd.DataFrame({'event_dt':event,'form':x.form.astype(str),'accession':x.get('accessionNumber')})
    out=out.dropna(subset=['event_dt']).drop_duplicates(['event_dt','form','accession']).sort_values('event_dt')
    out['event_dt']=out.event_dt.dt.tz_convert('America/New_York').dt.tz_localize(None)
    records=out.assign(event_dt=out.event_dt.astype(str)).to_dict('records'); p.write_text(json.dumps(records,separators=(',',':')))
    return out


def _days_since(snapshot:pd.Series,events:pd.Series)->np.ndarray:
    if len(events)==0:return np.full(len(snapshot),9999.0)
    ev=np.sort(pd.to_datetime(events).values.astype('datetime64[ns]')); sn=pd.to_datetime(snapshot).values.astype('datetime64[ns]'); pos=np.searchsorted(ev,sn,side='right')-1; out=np.full(len(sn),9999.0)
    good=pos>=0; out[good]=(sn[good]-ev[pos[good]])/np.timedelta64(1,'D'); return out.astype(float)


def add_symbol_event_features(frame:pd.DataFrame,filings:pd.DataFrame)->pd.DataFrame:
    x=frame.copy(); snap=pd.to_datetime(x.snapshot_dt)
    ff=filings.copy() if filings is not None else pd.DataFrame(columns=['event_dt','form'])
    if ff.empty:
        for c in EVENT_FEATURES:x[c]=0.0 if c.startswith(('sec_recent','sec_count')) else 9999.0
        return x
    ff['event_dt']=pd.to_datetime(ff.event_dt,errors='coerce'); ff=ff.dropna(subset=['event_dt'])
    groups={
        '8k':ff[ff.form.str.startswith('8-K')],
        '10q':ff[ff.form.str.startswith('10-Q')],
        '10k':ff[ff.form.str.startswith('10-K')],
        'major':ff,
    }
    for key,g in groups.items():x[f'sec_days_since_{key}']=_days_since(snap,g.event_dt)
    for key in ('8k','major'):
        d=x[f'sec_days_since_{key}']
        for days in (1,3,5,10):x[f'sec_recent_{key}_{days}d']=((d>=0)&(d<=days)).astype(float)
    # Count filings visible during the preceding 10 calendar days at each snapshot.
    ev=np.sort(ff.event_dt.values.astype('datetime64[ns]')); sn=snap.values.astype('datetime64[ns]'); hi=np.searchsorted(ev,sn,side='right'); lo=np.searchsorted(ev,sn-np.timedelta64(10,'D'),side='right'); x['sec_count_major_10d']=(hi-lo).astype(float)
    x['sec_post_periodic_5d']=((x.sec_days_since_10q<=5)|(x.sec_days_since_10k<=5)).astype(float)
    return x


EVENT_FEATURES=[
    'sec_days_since_8k','sec_days_since_10q','sec_days_since_10k','sec_days_since_major',
    'sec_recent_8k_1d','sec_recent_8k_3d','sec_recent_8k_5d','sec_recent_8k_10d',
    'sec_recent_major_1d','sec_recent_major_3d','sec_recent_major_5d','sec_recent_major_10d',
    'sec_count_major_10d','sec_post_periodic_5d']


def add_sec_features(data:pd.DataFrame,symbols:Iterable[str]|None=None,refresh:bool=False)->pd.DataFrame:
    parts=[]; wanted=set(str(s).upper() for s in symbols) if symbols is not None else set(data.symbol.astype(str).str.upper().unique())
    for symbol,g in data.groupby(data.symbol.astype(str).str.upper(),sort=False):
        if symbol not in wanted:
            parts.append(g); continue
        try:f=filings_for_symbol(symbol,refresh=refresh)
        except Exception:f=pd.DataFrame(columns=['event_dt','form'])
        parts.append(add_symbol_event_features(g,f))
    return pd.concat(parts,ignore_index=True).sort_values(['snapshot_dt','symbol']).reset_index(drop=True)
