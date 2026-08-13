from __future__ import annotations

import json, math, os, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"
OUT = Path("docs/data/latest.json")
FALLBACK = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","JPM","V","MA","LLY","WMT","COST","XOM","NFLX","ORCL","AMD","CRM","PLTR","UBER","NOW","SHOP","SPOT","PANW","CRWD","MU"]
SECRET_NAMES = ["TWELVE_DATA_API_KEY","FINNHUB_API_KEY","FMP_API_KEY","POLYGON_API_KEY","ALPHA_VANTAGE_API_KEY"]


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

def discover_universe():
    try:
        u=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",sep="|")
        u=u[(u["Test Issue"]=="N") & (u["ETF"]=="N")]
        syms=[]
        for raw in u["Symbol"].dropna().astype(str):
            s=raw.strip().replace(".","-")
            if s and len(s)<=7 and s.replace("-","").isalnum() and "$" not in s and " " not in s:
                syms.append(s)
        return sorted(set(syms))
    except Exception as e:
        print("Universe discovery failed; using fallback:",e)
        return FALLBACK

def market_regime(spy):
    c=spy.Close.dropna()
    if len(c)<200:return "UNKNOWN"
    s50=c.rolling(50).mean().iloc[-1]; s200=c.rolling(200).mean().iloc[-1]; last=c.iloc[-1]
    if last>s50>s200:return "RISK_ON"
    if last<s50<s200:return "RISK_OFF"
    return "MIXED"

def analyze(symbol,df,spy):
    df=df.dropna(subset=["Close","High","Low","Volume"]).copy(); spy=spy.dropna(subset=["Close"]).copy()
    if len(df)<210 or len(spy)<210:return None
    c=df.Close; v=df.Volume; last=float(c.iloc[-1])
    if last<2:return None
    sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean(); sma150=c.rolling(150).mean(); sma200=c.rolling(200).mean()
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean(); macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean()
    rs14=rsi(c); atr14=atr(df); ob=obv(df)
    p20=float(c.iloc[-21]); p63=float(c.iloc[-64]); p126=float(c.iloc[-127])
    ret20=last/p20-1; ret63=last/p63-1; ret126=last/p126-1
    s20=float(spy.Close.iloc[-1]/spy.Close.iloc[-21]-1); s63=float(spy.Close.iloc[-1]/spy.Close.iloc[-64]-1); s126=float(spy.Close.iloc[-1]/spy.Close.iloc[-127]-1)
    rs20=ret20-s20; rs63=ret63-s63; rs126=ret126-s126
    hi52=float(c.tail(252).max()); lo52=float(c.tail(252).min()); dist_high=last/hi52-1
    vol20=float(v.tail(20).mean()); dollar_vol=vol20*last
    if dollar_vol<5_000_000:return None
    vol_ratio=float(v.iloc[-1]/vol20) if vol20 else 0
    up=float(v[c.diff()>0].tail(20).sum()); down=float(v[c.diff()<0].tail(20).sum()); ud=up/down if down>0 else 2.0
    ob_slope=slope_pct(ob,20); persistence=float((c.pct_change().tail(20)>0).mean())
    macd_delta_pct=float((macd.iloc[-1]-sig.iloc[-1])/last*10000)
    atr_pct=float(atr14.iloc[-1]/last*100)

    trend=0
    trend += 18 if last>sma20.iloc[-1] else 0; trend += 22 if last>sma50.iloc[-1] else 0
    trend += 20 if sma50.iloc[-1]>sma150.iloc[-1] else 0; trend += 20 if sma150.iloc[-1]>sma200.iloc[-1] else 0
    trend += 20 if slope_pct(sma200,20)>0 else 0

    return {
      "symbol":symbol,"price":sf(last),"market_bar_date":pd.Timestamp(df.index[-1]).date().isoformat(),"change_1d_pct":sf(c.pct_change().iloc[-1]*100),
      "trend":round(trend),"rsi14":sf(rs14.iloc[-1]),"atr_pct":sf(atr_pct),"volume_ratio":sf(vol_ratio),"up_down_volume":sf(ud),
      "distance_52w_high_pct":sf(dist_high*100),"return_20d_pct":sf(ret20*100),"return_63d_pct":sf(ret63*100),"return_126d_pct":sf(ret126*100),
      "rs_20d_vs_spy_pct":sf(rs20*100),"rs_63d_vs_spy_pct":sf(rs63*100),"rs_126d_vs_spy_pct":sf(rs126*100),
      "sma20":sf(sma20.iloc[-1]),"sma50":sf(sma50.iloc[-1]),"sma150":sf(sma150.iloc[-1]),"sma200":sf(sma200.iloc[-1]),
      "high52":sf(hi52),"low52":sf(lo52),"anchor20":sf(p20),"anchor63":sf(p63),"anchor126":sf(p126),
      "spy_ret20":sf(s20),"spy_ret63":sf(s63),"spy_ret126":sf(s126),"macd_delta_pct":sf(macd_delta_pct),"persistence20":sf(persistence),
      "obv_slope_pct":sf(ob_slope),"avg_dollar_volume":sf(dollar_vol),"sma200_slope_up":slope_pct(sma200,20)>0,
    }

def pct_rank(s, ascending=True):
    x=pd.to_numeric(s,errors="coerce")
    return x.rank(pct=True,method="average",ascending=ascending).fillna(.5)*100

def recalibrate_rows(rows, regime):
    if not rows:return rows
    d=pd.DataFrame(rows)
    p20=pct_rank(d["return_20d_pct"]); p63=pct_rank(d["return_63d_pct"]); p126=pct_rank(d["return_126d_pct"])
    prs20=pct_rank(d["rs_20d_vs_spy_pct"]); prs63=pct_rank(d["rs_63d_vs_spy_pct"]); prs126=pct_rank(d["rs_126d_vs_spy_pct"])
    pmacd=pct_rank(d["macd_delta_pct"]); ppersist=pct_rank(d["persistence20"]); phigh=pct_rank(d["distance_52w_high_pct"])
    pvol=pct_rank(d["volume_ratio"]); pud=pct_rank(d["up_down_volume"]); pobv=pct_rank(d["obv_slope_pct"])
    pliq=pct_rank(np.log10(pd.to_numeric(d["avg_dollar_volume"],errors="coerce").clip(lower=1)))
    pvolq=100-pct_rank(d["atr_pct"])

    momentum=.18*p20+.24*p63+.14*p126+.12*prs20+.14*prs63+.06*prs126+.05*pmacd+.04*ppersist+.03*phigh
    relative=.25*prs20+.45*prs63+.30*prs126
    institutional=.30*d["trend"]+.18*relative+.14*pvol+.13*pud+.12*pobv+.08*phigh+.05*pliq
    factor_agreement=100-(pd.concat([momentum,relative,institutional,d["trend"]],axis=1).max(axis=1)-pd.concat([momentum,relative,institutional,d["trend"]],axis=1).min(axis=1))
    data_quality=.45*pliq+.30*pvolq+.25*factor_agreement.clip(lower=0)
    confidence=.55*factor_agreement.clip(lower=0)+.25*data_quality+.20*d["trend"]
    confluence=.30*momentum+.23*institutional+.15*d["trend"]+.17*relative+.15*confidence

    for i,row in enumerate(rows):
        m=float(momentum.iloc[i]); rel=float(relative.iloc[i]); inst=float(institutional.iloc[i]); conf=float(confidence.iloc[i]); con=float(confluence.iloc[i]); tr=float(row["trend"])
        bias="BULLISH" if tr>=60 and rel>=55 and m>=60 else "BEARISH" if tr<=40 and rel<=45 and m<=40 else "NEUTRAL"
        regime_alignment = (regime=="RISK_ON" and bias=="BULLISH") or (regime=="RISK_OFF" and bias=="BEARISH") or bias=="NEUTRAL" or regime in ("MIXED","UNKNOWN")
        if not regime_alignment:
            conf=max(0,conf-6); con=max(0,con-3)
        quality="A+" if con>=86 and conf>=78 else "A" if con>=78 and conf>=72 else "B" if con>=68 and conf>=64 else "C"
        signal="TOP DECILE MOMENTUM" if m>=90 and con>=80 else "STRONG MOMENTUM" if m>=75 and con>=68 else "WATCH" if con>=55 else "WEAK"
        reference=float(row["sma20"]) if bias!="NEUTRAL" else float(row["sma50"])
        evidence=[]
        if row["price"]>row["sma50"]>row["sma150"]>row["sma200"]:evidence.append("Bullish MA stack")
        if row["rs_63d_vs_spy_pct"]>5:evidence.append("Strong 3M relative strength")
        if row["volume_ratio"]>1.4 and row["change_1d_pct"]>0:evidence.append("Volume-backed advance")
        if row["up_down_volume"]>1.25:evidence.append("Accumulation-biased volume")
        if row["obv_slope_pct"]>0:evidence.append("OBV rising")
        if row["rsi14"] and row["rsi14"]>70:evidence.append("RSI extended")
        if row["atr_pct"] and row["atr_pct"]>5:evidence.append("High volatility risk")
        if regime_alignment:evidence.append("Aligned with market regime")
        row.update({
          "momentum":round(m),"confidence":round(conf),"confluence":round(con),"quality":quality,"bias":bias,"reference_level":sf(reference),
          "institutional_proxy":round(inst),"relative_strength":round(rel),"composite":round(con),"signal":signal,
          "factor_agreement":round(float(factor_agreement.iloc[i])),"data_quality":round(float(data_quality.iloc[i])),"market_regime":regime,
          "evidence":evidence[:8],"private_validation_available":False,
          "scoring_method":"Cross-sectional percentile ranks across the current liquid universe; avoids hard-score saturation."
        })
    return rows

def main():
    universe=discover_universe(); print("Discovered",len(universe),"US-listed non-ETF symbols")
    spy=yf.download(BENCHMARK,period="18mo",interval="1d",auto_adjust=True,progress=False)
    if isinstance(spy.columns,pd.MultiIndex): spy.columns=spy.columns.get_level_values(0)
    regime=market_regime(spy)
    rows=[]; chunk=160
    for i in range(0,len(universe),chunk):
        batch=universe[i:i+chunk]
        try:
            raw=yf.download(batch,period="18mo",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False,timeout=20)
            for s in batch:
                try:
                    df=raw[s] if isinstance(raw.columns,pd.MultiIndex) else raw
                    a=analyze(s,df,spy)
                    if a:rows.append(a)
                except Exception: pass
        except Exception as e: print("batch",i,"failed:",e)
        time.sleep(.25)
    rows=recalibrate_rows(rows,regime)
    rows.sort(key=lambda x:(x["confluence"],x["confidence"],x["momentum"]),reverse=True)
    market_dates=[r.get("market_bar_date") for r in rows if r.get("market_bar_date")]
    configured=[k for k in SECRET_NAMES if os.getenv(k)]
    payload={
      "generated_at":datetime.now(timezone.utc).isoformat(),"market_data_asof":max(market_dates) if market_dates else None,
      "market_data_type":"Daily adjusted OHLCV (1-day bars) + separate near-live multi-provider quote validation",
      "market_data_source":"Yahoo Finance via yfinance for historical bars; configured quote APIs validated separately",
      "configured_quote_providers":len(configured),"market_regime":regime,
      "scheduled_refresh":"Daily full-universe model after US close; near-live quote snapshots run separately during market hours",
      "benchmark":BENCHMARK,"discovered_count":len(universe),"universe_count":len(rows),
      "liquidity_filter":"Price >= $2 and 20-day average dollar volume >= $5M",
      "method":"Cross-sectional percentile model: momentum, relative strength, institutional proxy, trend, factor agreement and data quality. Scores are ranks/evidence, not calibrated win probabilities.",
      "stocks":rows
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,separators=(",",":")))
    print(f"wrote {len(rows)} liquid analyzed stocks to {OUT}; regime={regime}")

if __name__=="__main__": main()
