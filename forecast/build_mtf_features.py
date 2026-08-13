from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

LATEST = Path("docs/data/latest.json")
OUT = Path("docs/data/mtf_features.json")
TWELVE_KEY = os.getenv("TWELVE_DATA_API_KEY")
MAX_SYMBOLS = int(os.getenv("PHASE1_MTF_SYMBOLS", "40"))
TIMEOUT = 12
INTERVALS = {"15m": ("15min", 180), "4h": ("4h", 160), "day": ("1day", 220), "week": ("1week", 120)}


def get_json(url: str):
    req = Request(url, headers={"User-Agent": "Phase1-Forecast-Research/1.0"})
    with urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def normalize(node):
    vals = node.get("values") if isinstance(node, dict) else None
    if not isinstance(vals, list): return []
    rows=[]
    for r in vals:
        if not isinstance(r, dict): continue
        o,h,l,c=[num(r.get(k)) for k in ("open","high","low","close")]
        v=num(r.get("volume")) or 0.0
        if None in (o,h,l,c): continue
        rows.append({"datetime":r.get("datetime"),"open":o,"high":h,"low":l,"close":c,"volume":v})
    rows.sort(key=lambda x: str(x.get("datetime") or ""))
    return rows


def fetch_batch(symbols, interval, outputsize):
    if not TWELVE_KEY: raise RuntimeError("TWELVE_DATA_API_KEY is not configured")
    out, errors = {}, {}
    for i in range(0, len(symbols), 8):
        batch=symbols[i:i+8]
        q=urlencode({"symbol":",".join(batch),"interval":interval,"outputsize":outputsize,"apikey":TWELVE_KEY})
        try:
            d=get_json("https://api.twelvedata.com/time_series?"+q)
            if len(batch)==1 and isinstance(d,dict) and "values" in d:
                vals=normalize(d)
                if vals: out[batch[0]]=vals
                else: errors[batch[0]]=str(d.get("message") or "no values")
            else:
                for s in batch:
                    node=d.get(s) if isinstance(d,dict) else None
                    vals=normalize(node)
                    if vals: out[s]=vals
                    else: errors[s]=str(node.get("message") if isinstance(node,dict) else "no values")
        except Exception as e:
            for s in batch: errors[s]=type(e).__name__
        time.sleep(.25)
    return out, errors


def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return 100-(100/(1+rs))


def features(rows):
    if len(rows)<55: return None
    d=pd.DataFrame(rows)
    for k in ["open","high","low","close","volume"]: d[k]=pd.to_numeric(d[k],errors="coerce")
    d=d.dropna(subset=["open","high","low","close"])
    if len(d)<55: return None
    c=d.close; v=d.volume.fillna(0)
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean(); macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean()
    rs=rsi(c); prev=c.shift(1); tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean()
    vol20=v.rolling(20).mean(); typical=(d.high+d.low+d.close)/3; rvwap=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    last=float(c.iloc[-1]); ret3=last/c.iloc[-4]-1; ret10=last/c.iloc[-11]-1
    vol_ratio=float(v.iloc[-1]/vol20.iloc[-1]) if math.isfinite(vol20.iloc[-1]) and vol20.iloc[-1] else None
    return {"last_bar":rows[-1].get("datetime"),"close":round(last,4),"return_3bars_pct":round(ret3*100,3),"return_10bars_pct":round(ret10*100,3),"rsi14":round(float(rs.iloc[-1]),2) if math.isfinite(rs.iloc[-1]) else None,"ema20":round(float(ema20.iloc[-1]),4),"ema50":round(float(ema50.iloc[-1]),4),"macd_delta_pct":round(float((macd.iloc[-1]-sig.iloc[-1])/last*100),5),"atr_pct":round(float(atr.iloc[-1]/last*100),3) if math.isfinite(atr.iloc[-1]) else None,"volume_ratio20":round(vol_ratio,3) if vol_ratio is not None else None,"rolling_vwap20_distance_pct":round(float((last/rvwap.iloc[-1]-1)*100),3) if math.isfinite(rvwap.iloc[-1]) and rvwap.iloc[-1] else None,"above_ema20":bool(last>ema20.iloc[-1]),"ema20_above_ema50":bool(ema20.iloc[-1]>ema50.iloc[-1])}


def main():
    latest=json.loads(LATEST.read_text())
    stocks=[]
    for x in latest.get("stocks",[]):
        p=num(x.get("price")); dv=num(x.get("avg_dollar_volume"))
        if p is None or dv is None or p<5 or dv<20_000_000: continue
        y=dict(x); y["_dv"]=dv; stocks.append(y)
    stocks=sorted(stocks,key=lambda x:x["_dv"],reverse=True)[:MAX_SYMBOLS]
    symbols=[x["symbol"] for x in stocks]
    matrix={s:{} for s in symbols}; errors={s:{} for s in symbols}
    for label,(interval,size) in INTERVALS.items():
        data,err=fetch_batch(symbols,interval,size)
        for s in symbols:
            f=features(data.get(s,[]))
            if f: matrix[s][label]=f
            else: errors[s][label]=err.get(s,"insufficient bars")
    rows=[]
    for x in stocks:
        s=x["symbol"]; coverage=len(matrix[s])/len(INTERVALS)
        rows.append({"symbol":s,"price":x.get("price"),"avg_dollar_volume":x.get("avg_dollar_volume"),"feature_coverage_pct":round(coverage*100,1),"forecast_eligible":coverage>=0.75,"timeframes":matrix[s],"errors":errors[s]})
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"RESEARCH FEATURE DATASET — NO FORECAST PROBABILITIES","framework":"15m + 4h + Day + Week; live quote remains in intraday.json","selection_rule":"Most liquid current Phase1 names with price >= $5 and 20-day average dollar volume >= $20M; selection is for data collection, not a directional recommendation.","symbols_requested":len(symbols),"symbols_with_75pct_coverage":sum(1 for r in rows if r["forecast_eligible"]),"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,separators=(",",":")))
    print("wrote",len(rows),"multi-timeframe research rows")


if __name__=="__main__": main()
