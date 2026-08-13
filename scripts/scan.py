from __future__ import annotations

import json, math, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","JPM","V","MA","LLY","WMT","COST","XOM","UNH","NFLX","ORCL","HD","BAC","AMD","CRM","KO","PEP","ADBE","CSCO","ABT","MRK","CVX","MCD","TMO","ACN","LIN","QCOM","TXN","INTU","AMAT","GE","CAT","GS","UBER","NOW","PLTR","SHOP","SPOT","PANW","CRWD","ARM","MU","SMCI"]
BENCHMARK = "SPY"
OUT = Path("docs/data/latest.json")


def clamp(x, lo=0, hi=100): return max(lo, min(hi, x))

def sf(x):
    try:
        x=float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x,4)
    except Exception: return None

def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return 100-(100/(1+rs))

def atr(df,n=14):
    p=df.Close.shift(1)
    tr=pd.concat([(df.High-df.Low),(df.High-p).abs(),(df.Low-p).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv(df): return (np.sign(df.Close.diff()).fillna(0)*df.Volume).cumsum()

def slope_pct(s,lookback=20):
    s=s.dropna().tail(lookback)
    if len(s)<10:return 0.0
    x=np.arange(len(s),dtype=float); y=s.values.astype(float); m=np.nanmean(np.abs(y))
    return 0.0 if not m else float(np.polyfit(x,y,1)[0]/m*100)

def analyze(symbol,df,spy):
    df=df.dropna(subset=["Close","High","Low","Volume"]).copy(); spy=spy.dropna(subset=["Close"]).copy()
    if len(df)<210 or len(spy)<210:return None
    c=df.Close; v=df.Volume; last=float(c.iloc[-1])
    sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean(); sma150=c.rolling(150).mean(); sma200=c.rolling(200).mean()
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean(); macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean()
    rs14=rsi(c); atr14=atr(df); ob=obv(df)
    ret20=last/float(c.iloc[-21])-1; ret63=last/float(c.iloc[-64])-1; ret126=last/float(c.iloc[-127])-1
    s20=float(spy.Close.iloc[-1]/spy.Close.iloc[-21]-1); s63=float(spy.Close.iloc[-1]/spy.Close.iloc[-64]-1); s126=float(spy.Close.iloc[-1]/spy.Close.iloc[-127]-1)
    rs20=ret20-s20; rs63=ret63-s63; rs126=ret126-s126
    hi52=float(c.tail(252).max()); lo52=float(c.tail(252).min()); dist_high=last/hi52-1
    vol20=float(v.tail(20).mean()); dollar_vol=vol20*last; vol_ratio=float(v.iloc[-1]/vol20) if vol20 else 0
    up=float(v[c.diff()>0].tail(20).sum()); down=float(v[c.diff()<0].tail(20).sum()); ud=up/down if down>0 else 2.0
    ob_slope=slope_pct(ob,20); persistence=float((c.pct_change().tail(20)>0).mean())

    trend=0
    trend += 18 if last>sma20.iloc[-1] else 0; trend += 22 if last>sma50.iloc[-1] else 0
    trend += 20 if sma50.iloc[-1]>sma150.iloc[-1] else 0; trend += 20 if sma150.iloc[-1]>sma200.iloc[-1] else 0
    trend += 20 if slope_pct(sma200,20)>0 else 0

    relative=clamp(50+rs20*180+rs63*120+rs126*70)
    momentum=clamp(22+ret20*210+ret63*120+ret126*55+(float(rs14.iloc[-1])-50)*0.55+(float(macd.iloc[-1]-sig.iloc[-1])/last*10000)+persistence*18)
    breakout=clamp((1+dist_high)*100)*.35+clamp(vol_ratio*45)*.25+clamp((ud-.7)*70)*.2+clamp(50+ob_slope*120)*.2
    institutional=clamp(trend*.28+relative*.24+breakout*.24+clamp(50+ob_slope*150)*.12+clamp((ud-.5)*65)*.12)

    atr_pct=float(atr14.iloc[-1]/last*100); liquidity=clamp((math.log10(max(dollar_vol,1))-5.5)*28)
    has_private=any(os.getenv(k) for k in ["TWELVE_DATA_API_KEY","FINNHUB_API_KEY","FMP_API_KEY","POLYGON_API_KEY","ALPHA_VANTAGE_API_KEY"])
    validation=92 if has_private else 70; vol_quality=clamp(100-max(0,atr_pct-2.2)*12); agreement=100-abs(momentum-institutional)*.55
    confidence=clamp(100*.2+validation*.2+liquidity*.15+vol_quality*.15+agreement*.2+trend*.1)
    composite=clamp(momentum*.33+institutional*.31+trend*.16+relative*.20)
    label="HIGH CONVICTION" if composite>=80 and confidence>=75 else "STRONG" if composite>=68 and confidence>=65 else "WATCH" if composite>=55 else "WEAK"
    evidence=[]
    if last>sma50.iloc[-1]>sma150.iloc[-1]>sma200.iloc[-1]: evidence.append("Bullish MA stack")
    if rs63>.05:evidence.append("Strong 3M relative strength")
    if vol_ratio>1.4 and c.iloc[-1]>c.iloc[-2]:evidence.append("Volume-backed advance")
    if ud>1.25:evidence.append("Accumulation-biased volume")
    if ob_slope>0:evidence.append("OBV rising")
    if float(rs14.iloc[-1])>70:evidence.append("RSI extended")
    if atr_pct>5:evidence.append("High volatility risk")
    if not has_private:evidence.append("No private cross-source validation")
    return {"symbol":symbol,"price":sf(last),"change_1d_pct":sf(c.pct_change().iloc[-1]*100),"momentum":round(momentum),"confidence":round(confidence),"institutional_proxy":round(institutional),"trend":round(trend),"relative_strength":round(relative),"composite":round(composite),"signal":label,"rsi14":sf(rs14.iloc[-1]),"atr_pct":sf(atr_pct),"volume_ratio":sf(vol_ratio),"up_down_volume":sf(ud),"distance_52w_high_pct":sf(dist_high*100),"return_20d_pct":sf(ret20*100),"return_63d_pct":sf(ret63*100),"rs_63d_vs_spy_pct":sf(rs63*100),"sma20":sf(sma20.iloc[-1]),"sma50":sf(sma50.iloc[-1]),"sma150":sf(sma150.iloc[-1]),"sma200":sf(sma200.iloc[-1]),"evidence":evidence[:6],"private_validation_available":has_private}

def main():
    tickers=UNIVERSE+[BENCHMARK]
    raw=yf.download(tickers,period="18mo",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
    spy=raw[BENCHMARK] if isinstance(raw.columns,pd.MultiIndex) else raw
    rows=[]
    for s in UNIVERSE:
        try:
            df=raw[s] if isinstance(raw.columns,pd.MultiIndex) else raw
            a=analyze(s,df,spy)
            if a:rows.append(a)
        except Exception as e: print(f"{s}: {e}")
    rows.sort(key=lambda x:(x["momentum"],x["confidence"],x["composite"]),reverse=True)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"benchmark":BENCHMARK,"universe_count":len(rows),"method":"Momentum + trend + benchmark RS + observable institutional proxies; confidence is separate and data-quality aware.","stocks":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2))
    print(f"wrote {len(rows)} rows to {OUT}")

if __name__=="__main__": main()
