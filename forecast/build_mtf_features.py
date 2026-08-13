from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LATEST = Path("docs/data/latest.json")
OUT = Path("docs/data/mtf_features.json")
MAX_SYMBOLS = int(os.getenv("PHASE1_MTF_SYMBOLS", "40"))


def num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def rows_from_df(df: pd.DataFrame):
    if df is None or df.empty: return []
    d=df.copy()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    need=["Open","High","Low","Close","Volume"]
    if any(k not in d.columns for k in need): return []
    d=d.dropna(subset=["Open","High","Low","Close"])
    out=[]
    for idx,r in d.iterrows():
        out.append({"datetime":str(idx),"open":float(r.Open),"high":float(r.High),"low":float(r.Low),"close":float(r.Close),"volume":float(r.Volume) if pd.notna(r.Volume) else 0.0})
    return out


def bulk_download(symbols, period, interval):
    result={s:[] for s in symbols}
    for i in range(0,len(symbols),20):
        batch=symbols[i:i+20]
        try:
            raw=yf.download(batch,period=period,interval=interval,auto_adjust=True,group_by="ticker",threads=True,progress=False,timeout=30,prepost=False)
            for s in batch:
                try:
                    df=raw[s] if isinstance(raw.columns,pd.MultiIndex) else raw
                    result[s]=rows_from_df(df)
                except Exception:
                    result[s]=[]
        except Exception:
            pass
        time.sleep(.7)
    return result


def aggregate_4h(rows):
    if not rows: return []
    d=pd.DataFrame(rows)
    d["dt"]=pd.to_datetime(d["datetime"],errors="coerce")
    d=d.dropna(subset=["dt"]).sort_values("dt")
    if d.empty:return []
    out=[]
    for _,g in d.groupby(d["dt"].dt.date):
        g=g.sort_values("dt").reset_index(drop=True)
        for i in range(0,len(g),4):
            b=g.iloc[i:i+4]
            if b.empty:continue
            out.append({"datetime":str(b.dt.iloc[-1]),"open":float(b.open.iloc[0]),"high":float(b.high.max()),"low":float(b.low.min()),"close":float(b.close.iloc[-1]),"volume":float(b.volume.sum())})
    return out


def aggregate_week(rows):
    if not rows:return []
    d=pd.DataFrame(rows)
    d["dt"]=pd.to_datetime(d["datetime"],errors="coerce");d=d.dropna(subset=["dt"]).set_index("dt").sort_index()
    if d.empty:return []
    w=pd.DataFrame({"open":d.open.resample("W-FRI").first(),"high":d.high.resample("W-FRI").max(),"low":d.low.resample("W-FRI").min(),"close":d.close.resample("W-FRI").last(),"volume":d.volume.resample("W-FRI").sum()}).dropna(subset=["open","high","low","close"])
    return [{"datetime":str(idx),"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close),"volume":float(r.volume)} for idx,r in w.iterrows()]


def rsi(c,n=14):
    d=c.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-(100/(1+rs))


def features(rows):
    if len(rows)<55:return None
    d=pd.DataFrame(rows)
    for k in ["open","high","low","close","volume"]:d[k]=pd.to_numeric(d[k],errors="coerce")
    d=d.dropna(subset=["open","high","low","close"])
    if len(d)<55:return None
    c=d.close;v=d.volume.fillna(0);ema20=c.ewm(span=20,adjust=False).mean();ema50=c.ewm(span=50,adjust=False).mean();ema12=c.ewm(span=12,adjust=False).mean();ema26=c.ewm(span=26,adjust=False).mean();macd=ema12-ema26;sig=macd.ewm(span=9,adjust=False).mean();rs=rsi(c)
    prev=c.shift(1);tr=pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1);atr=tr.rolling(14).mean();vol20=v.rolling(20).mean();typical=(d.high+d.low+d.close)/3;rvwap=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    last=float(c.iloc[-1]);ret3=last/c.iloc[-4]-1;ret10=last/c.iloc[-11]-1;vr=float(v.iloc[-1]/vol20.iloc[-1]) if math.isfinite(vol20.iloc[-1]) and vol20.iloc[-1] else None
    return {"last_bar":rows[-1].get("datetime"),"close":round(last,4),"return_3bars_pct":round(ret3*100,3),"return_10bars_pct":round(ret10*100,3),"rsi14":round(float(rs.iloc[-1]),2) if math.isfinite(rs.iloc[-1]) else None,"ema20":round(float(ema20.iloc[-1]),4),"ema50":round(float(ema50.iloc[-1]),4),"macd_delta_pct":round(float((macd.iloc[-1]-sig.iloc[-1])/last*100),5),"atr_pct":round(float(atr.iloc[-1]/last*100),3) if math.isfinite(atr.iloc[-1]) else None,"volume_ratio20":round(vr,3) if vr is not None else None,"rolling_vwap20_distance_pct":round(float((last/rvwap.iloc[-1]-1)*100),3) if math.isfinite(rvwap.iloc[-1]) and rvwap.iloc[-1] else None,"above_ema20":bool(last>ema20.iloc[-1]),"ema20_above_ema50":bool(ema20.iloc[-1]>ema50.iloc[-1])}


def select_symbols(stocks):
    eligible=[]
    for x in stocks:
        p=num(x.get("price"));dv=num(x.get("avg_dollar_volume"))
        if p is None or dv is None or p<5 or dv<20_000_000:continue
        y=dict(x);y["_dv"]=dv;y["_activity"]=abs(num(x.get("change_1d_pct")) or 0)*2+abs(num(x.get("rs_63d_vs_spy_pct")) or 0)+abs((num(x.get("volume_ratio")) or 1)-1)*5;eligible.append(y)
    liquid=sorted(eligible,key=lambda x:x["_dv"],reverse=True)[:max(1,MAX_SYMBOLS//2)]
    active=sorted(eligible,key=lambda x:x["_activity"],reverse=True)[:MAX_SYMBOLS]
    out=[];seen=set()
    for x in liquid+active:
        if x["symbol"] in seen:continue
        seen.add(x["symbol"]);out.append(x)
        if len(out)>=MAX_SYMBOLS:break
    return out


def main():
    latest=json.loads(LATEST.read_text());stocks=select_symbols(latest.get("stocks",[]));symbols=[x["symbol"] for x in stocks]
    bars15=bulk_download(symbols,"60d","15m");bars1h=bulk_download(symbols,"730d","1h");barsday=bulk_download(symbols,"5y","1d")
    rows=[]
    for x in stocks:
        s=x["symbol"]
        tf={"15m":features(bars15.get(s,[])),"4h":features(aggregate_4h(bars1h.get(s,[]))),"day":features(barsday.get(s,[])),"week":features(aggregate_week(barsday.get(s,[])))}
        tf={k:v for k,v in tf.items() if v};coverage=len(tf)/4
        rows.append({"symbol":s,"price":x.get("price"),"avg_dollar_volume":x.get("avg_dollar_volume"),"feature_coverage_pct":round(coverage*100,1),"forecast_eligible":coverage>=0.75,"timeframes":tf,"errors":{k:"insufficient bulk bars" for k in ["15m","4h","day","week"] if k not in tf}})
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"RESEARCH FEATURE DATASET — NO FORECAST PROBABILITIES","framework":"15m + 4h + Day + Week; current quote validation remains separate in intraday.json","sources":"Bulk adjusted historical bars for broad feature coverage; scarce real-time API credits reserved for final candidate validation.","selection_rule":"Liquidity plus absolute market activity for research coverage only; not a directional recommendation.","symbols_requested":len(symbols),"symbols_with_75pct_coverage":sum(1 for r in rows if r["forecast_eligible"]),"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,separators=(",",":")));print("wrote",len(rows),"rows; eligible",payload["symbols_with_75pct_coverage"])


if __name__=="__main__":main()
