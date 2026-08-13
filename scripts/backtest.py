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
OUT = Path("docs/data/backtest.json")
BENCHMARK = "SPY"
SAMPLE_STEP = int(os.getenv("PHASE1_BACKTEST_SAMPLE_STEP", "21"))  # about monthly; reduces overlapping 20D outcomes
MAX_SYMBOLS = int(os.getenv("PHASE1_BACKTEST_SYMBOLS", "0"))  # 0 = all symbols in current liquid-universe snapshot
CHUNK = int(os.getenv("PHASE1_BACKTEST_CHUNK", "45"))


def sf(v, digits=4):
    try:
        x = float(v)
        if not math.isfinite(x):
            return None
        return round(x, digits)
    except Exception:
        return None


def rsi(c: pd.Series, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n=14):
    p = df["Close"].shift(1)
    tr = pd.concat([(df["High"] - df["Low"]), (df["High"] - p).abs(), (df["Low"] - p).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rolling_slope_pct(s: pd.Series, n=20):
    """Fast rolling OLS slope divided by mean absolute level, matching scan.py semantics closely."""
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return pd.Series(out, index=s.index)
    w = np.arange(n, dtype=float)
    sx = w.sum()
    sx2 = np.square(w).sum()
    denom = n * sx2 - sx * sx
    for i in range(n - 1, len(a)):
        y = a[i - n + 1 : i + 1]
        if not np.isfinite(y).all():
            continue
        sy = y.sum()
        sxy = np.dot(w, y)
        slope = (n * sxy - sx * sy) / denom
        mean_abs = np.mean(np.abs(y))
        out[i] = 0.0 if mean_abs == 0 else slope / mean_abs * 100
    return pd.Series(out, index=s.index)


def build_spy():
    spy = yf.download(BENCHMARK, period="max", interval="1d", auto_adjust=True, progress=False, timeout=30)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy.dropna(subset=["Close"]).copy()
    c = spy["Close"]
    spy["spy20"] = c / c.shift(20) - 1
    spy["spy63"] = c / c.shift(63) - 1
    spy["spy126"] = c / c.shift(126) - 1
    spy["sma50"] = c.rolling(50).mean()
    spy["sma200"] = c.rolling(200).mean()
    spy["regime"] = np.where((c > spy["sma50"]) & (spy["sma50"] > spy["sma200"]), "RISK_ON",
                      np.where((c < spy["sma50"]) & (spy["sma50"] < spy["sma200"]), "RISK_OFF", "MIXED"))
    for h in (5, 10, 20):
        spy[f"spy_fwd_{h}"] = c.shift(-h) / c - 1
    return spy


def symbol_features(symbol: str, df: pd.DataFrame, spy: pd.DataFrame, sample_dates: pd.DatetimeIndex):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Close", "High", "Low", "Volume"]
    if any(k not in df.columns for k in need):
        return None
    df = df.dropna(subset=need).copy()
    if len(df) < 280:
        return None
    c, v = df["Close"], df["Volume"]
    sma20 = c.rolling(20).mean(); sma50 = c.rolling(50).mean(); sma150 = c.rolling(150).mean(); sma200 = c.rolling(200).mean()
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean(); macd = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean()
    atr14 = atr(df)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    ob_slope = rolling_slope_pct(obv, 20)
    sma200_slope = rolling_slope_pct(sma200, 20)

    ret20 = c / c.shift(20) - 1; ret63 = c / c.shift(63) - 1; ret126 = c / c.shift(126) - 1
    high52 = c.rolling(252, min_periods=200).max()
    vol20 = v.rolling(20).mean(); dollar_vol = vol20 * c; vol_ratio = v / vol20
    posvol = v.where(c.diff() > 0, 0.0).rolling(20).sum(); negvol = v.where(c.diff() < 0, 0.0).rolling(20).sum()
    ud = posvol / negvol.replace(0, np.nan)
    persistence = (c.pct_change() > 0).astype(float).rolling(20).mean()
    macd_delta = (macd - sig) / c * 10000
    atr_pct = atr14 / c * 100

    trend = (
        (c > sma20).astype(float) * 18 +
        (c > sma50).astype(float) * 22 +
        (sma50 > sma150).astype(float) * 20 +
        (sma150 > sma200).astype(float) * 20 +
        (sma200_slope > 0).astype(float) * 20
    )

    f = pd.DataFrame(index=df.index)
    f["symbol"] = symbol
    f["price"] = c
    f["ret20"] = ret20 * 100; f["ret63"] = ret63 * 100; f["ret126"] = ret126 * 100
    f["dist_high"] = (c / high52 - 1) * 100
    f["vol_ratio"] = vol_ratio; f["ud"] = ud.clip(upper=8); f["obv_slope"] = ob_slope
    f["persistence"] = persistence; f["macd_delta"] = macd_delta; f["atr_pct"] = atr_pct
    f["avg_dollar_volume"] = dollar_vol; f["trend"] = trend
    for h in (5, 10, 20):
        f[f"fwd_{h}"] = (c.shift(-h) / c - 1) * 100

    s = spy.reindex(f.index)
    f["rs20"] = (ret20 - s["spy20"]) * 100
    f["rs63"] = (ret63 - s["spy63"]) * 100
    f["rs126"] = (ret126 - s["spy126"]) * 100
    f["regime"] = s["regime"]
    for h in (5, 10, 20):
        f[f"spy_fwd_{h}"] = s[f"spy_fwd_{h}"] * 100

    idx = f.index.intersection(sample_dates)
    f = f.loc[idx]
    f = f[(f["price"] >= 2) & (f["avg_dollar_volume"] >= 5_000_000)]
    f = f.dropna(subset=["ret20", "ret63", "ret126", "rs20", "rs63", "rs126", "macd_delta", "persistence", "dist_high", "atr_pct"])
    return f.reset_index(names="date") if len(f) else None


def pct_rank_group(d: pd.DataFrame, col: str, ascending=True):
    return d.groupby("date")[col].rank(pct=True, method="average", ascending=ascending).fillna(0.5) * 100


def score(records: pd.DataFrame):
    d = records.copy()
    p20 = pct_rank_group(d, "ret20"); p63 = pct_rank_group(d, "ret63"); p126 = pct_rank_group(d, "ret126")
    prs20 = pct_rank_group(d, "rs20"); prs63 = pct_rank_group(d, "rs63"); prs126 = pct_rank_group(d, "rs126")
    pmacd = pct_rank_group(d, "macd_delta"); ppersist = pct_rank_group(d, "persistence"); phigh = pct_rank_group(d, "dist_high")
    pvol = pct_rank_group(d, "vol_ratio"); pud = pct_rank_group(d, "ud"); pobv = pct_rank_group(d, "obv_slope")
    d["logliq"] = np.log10(pd.to_numeric(d["avg_dollar_volume"], errors="coerce").clip(lower=1))
    pliq = pct_rank_group(d, "logliq"); pvolq = 100 - pct_rank_group(d, "atr_pct")

    d["momentum"] = .18*p20 + .24*p63 + .14*p126 + .12*prs20 + .14*prs63 + .06*prs126 + .05*pmacd + .04*ppersist + .03*phigh
    d["relative"] = .25*prs20 + .45*prs63 + .30*prs126
    d["institutional"] = .30*d["trend"] + .18*d["relative"] + .14*pvol + .13*pud + .12*pobv + .08*phigh + .05*pliq
    mx = pd.concat([d["momentum"], d["relative"], d["institutional"], d["trend"]], axis=1).max(axis=1)
    mn = pd.concat([d["momentum"], d["relative"], d["institutional"], d["trend"]], axis=1).min(axis=1)
    d["factor_agreement"] = (100 - (mx - mn)).clip(lower=0)
    d["data_quality"] = .45*pliq + .30*pvolq + .25*d["factor_agreement"]
    d["confidence"] = .55*d["factor_agreement"] + .25*d["data_quality"] + .20*d["trend"]
    d["confluence"] = .30*d["momentum"] + .23*d["institutional"] + .15*d["trend"] + .17*d["relative"] + .15*d["confidence"]

    bullish = (d["trend"] >= 60) & (d["relative"] >= 55) & (d["momentum"] >= 60)
    bearish = (d["trend"] <= 40) & (d["relative"] <= 45) & (d["momentum"] <= 40)
    d["bias"] = np.where(bullish, "BULLISH", np.where(bearish, "BEARISH", "NEUTRAL"))
    misaligned = ((d["regime"] == "RISK_ON") & (d["bias"] == "BEARISH")) | ((d["regime"] == "RISK_OFF") & (d["bias"] == "BULLISH"))
    d.loc[misaligned, "confidence"] = (d.loc[misaligned, "confidence"] - 6).clip(lower=0)
    d.loc[misaligned, "confluence"] = (d.loc[misaligned, "confluence"] - 3).clip(lower=0)
    d["confluence_bucket"] = pd.cut(d["confluence"], bins=[-1, 60, 70, 80, 90, 101], labels=["<60", "60-69", "70-79", "80-89", "90+"] , right=False)
    d["momentum_bucket"] = pd.cut(d["momentum"], bins=[-1, 60, 70, 80, 90, 101], labels=["<60", "60-69", "70-79", "80-89", "90+"], right=False)
    return d


def summarize_group(d: pd.DataFrame, group_col: str):
    out = {}
    for key, g in d.groupby(group_col, observed=True):
        k = str(key)
        out[k] = {}
        for h in (5, 10, 20):
            x = pd.to_numeric(g[f"fwd_{h}"], errors="coerce").dropna()
            aligned = g.loc[x.index]
            spy = pd.to_numeric(aligned[f"spy_fwd_{h}"], errors="coerce")
            excess = x - spy
            out[k][f"{h}d"] = {
                "n": int(len(x)),
                "positive_rate_pct": sf((x > 0).mean() * 100, 2),
                "beat_spy_rate_pct": sf((excess > 0).mean() * 100, 2),
                "median_return_pct": sf(x.median(), 3),
                "mean_return_pct": sf(x.mean(), 3),
                "median_excess_vs_spy_pct": sf(excess.median(), 3),
                "p25_return_pct": sf(x.quantile(.25), 3),
                "p75_return_pct": sf(x.quantile(.75), 3),
            }
    return out


def main():
    latest = json.loads(LATEST.read_text())
    current = latest.get("stocks", [])
    current = sorted(current, key=lambda x: x.get("avg_dollar_volume", 0) or 0, reverse=True)
    symbols = [x["symbol"] for x in current]
    if MAX_SYMBOLS > 0:
        symbols = symbols[:MAX_SYMBOLS]
    print("backtest universe", len(symbols), "current liquid symbols")

    spy = build_spy()
    sample_dates = spy.index[260::SAMPLE_STEP]
    frames = []
    attempted = 0
    with_history = 0

    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i:i + CHUNK]
        attempted += len(batch)
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(batch, period="max", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False, timeout=40)
                break
            except Exception as e:
                print("download batch", i, "attempt", attempt + 1, "failed", type(e).__name__)
                time.sleep(4 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for s in batch:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                f = symbol_features(s, df, spy, sample_dates)
                if f is not None and len(f):
                    frames.append(f)
                    with_history += 1
            except Exception as e:
                print("feature failed", s, type(e).__name__)
        if i and i % (CHUNK * 5) == 0:
            print("processed", attempted, "symbols; usable", with_history)
        time.sleep(1.2)

    if not frames:
        raise RuntimeError("No historical observations produced")
    records = pd.concat(frames, ignore_index=True)
    scored = score(records)
    scored = scored.dropna(subset=["fwd_5", "fwd_10", "fwd_20"])
    dates = pd.Series(pd.to_datetime(scored["date"]).sort_values().unique())
    split_idx = max(1, int(len(dates) * .8))
    holdout_start = pd.Timestamp(dates.iloc[split_idx]) if split_idx < len(dates) else pd.Timestamp(dates.iloc[-1])
    holdout = scored[pd.to_datetime(scored["date"]) >= holdout_start]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "Phase1 cross-sectional percentile momentum/confluence model",
        "history_source": "Yahoo Finance via yfinance, adjusted daily OHLCV, period=max",
        "benchmark": BENCHMARK,
        "sampling": f"Every {SAMPLE_STEP} SPY trading days after 260-day warmup; approximately monthly to reduce overlapping 20-day outcomes",
        "horizons_trading_days": [5, 10, 20],
        "universe_definition": "Current Phase1 liquid-universe symbols, filtered again point-in-time by price >= $2 and 20-day average dollar volume >= $5M",
        "survivorship_bias_warning": "Yes. Historical delisted/failed companies are not reconstructed because the universe starts from today's listed symbols. Treat results as calibration of current survivors, not a fully point-in-time investable-universe backtest.",
        "lookahead_control": "All score inputs use data at or before each snapshot date. Forward returns are used only as outcomes. Model weights are not optimized by this script.",
        "symbols_requested": len(symbols),
        "symbols_with_usable_history": with_history,
        "observations": int(len(scored)),
        "first_snapshot": pd.Timestamp(scored["date"].min()).date().isoformat(),
        "last_snapshot": pd.Timestamp(scored["date"].max()).date().isoformat(),
        "holdout_start": holdout_start.date().isoformat(),
        "full_sample_by_confluence": summarize_group(scored, "confluence_bucket"),
        "holdout_by_confluence": summarize_group(holdout, "confluence_bucket"),
        "full_sample_by_momentum": summarize_group(scored, "momentum_bucket"),
        "full_sample_by_regime": summarize_group(scored, "regime"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote", OUT, "observations", len(scored), "history", payload["first_snapshot"], "to", payload["last_snapshot"])


if __name__ == "__main__":
    main()
