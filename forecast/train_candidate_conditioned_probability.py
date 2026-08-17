from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_MODEL = Path("forecast/data/candidate_conditioned_probability.joblib")
OUT_VALIDATION = Path("docs/data/candidate_conditioned_validation.json")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_VALIDATION.parent.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10]
MIN_PRICE = 5.0
MIN_ADV20 = 100_000_000.0
FINAL_START = pd.Timestamp("2026-01-01")

SECTOR_GROUPS = {
    "XLK": "AAPL MSFT NVDA AMD AVGO ORCL CRM ADBE CSCO INTC QCOM AMAT MU TXN IBM NOW PANW CRWD SNOW PLTR".split(),
    "XLC": "META GOOGL GOOG NFLX DIS CMCSA T VZ".split(),
    "XLY": "AMZN TSLA HD LOW NKE MCD SBUX BKNG MAR ABNB UBER TGT GM F".split(),
    "XLF": "JPM BAC WFC C GS MS SCHW V MA AXP COF BLK SPGI CME".split(),
    "XLV": "LLY UNH JNJ ABBV MRK PFE TMO AMGN GILD ISRG MDT CVS".split(),
    "XLE": "XOM CVX COP SLB EOG OXY MPC VLO".split(),
    "XLI": "CAT DE GE RTX BA HON UPS FDX LMT NOC ETN MMM".split(),
    "XLP": "WMT COST KO PEP PM MO MDLZ CL PG".split(),
    "XLU": "NEE SO DUK AEP EXC SRE".split(),
    "XLB": "LIN APD FCX NUE DOW".split(),
    "XLRE": "AMT PLD EQIX SPG O".split(),
}
SECTOR_OF = {s: etf for etf, names in SECTOR_GROUPS.items() for s in names}
UNIVERSE = sorted(SECTOR_OF)
BENCHMARKS = ["SPY", "QQQ", "IWM", "^VIX", "TLT", "GLD", "USO"] + list(SECTOR_GROUPS)

FEATURES = [
    "candidate_direction_score", "candidate_activity", "candidate_rank_abs", "candidate_side_code",
    "ret1", "ret5", "ret20", "ret63", "rsi14", "above_ema20", "ema20_gt_ema50", "ema50_gt_ema200",
    "macd_delta_pct", "atr_pct", "rv20", "volume_ratio20", "adv20_log", "range_pct", "gap_pct",
    "close_open_pct", "pos_52w", "rel20_spy", "rel63_spy", "sector_rel20", "sector_rel63",
    "weekly_ret1", "weekly_ret4", "weekly_above_ema10", "weekly_ema10_gt_ema20",
    "spy_ret5", "spy_ret20", "spy_ret63", "spy_above_ema50", "spy_ema50_gt_ema200",
    "qqq_rel20_spy", "iwm_rel20_spy", "vix_level", "vix_z20", "vix_ret5",
    "tlt_ret20", "gld_ret20", "uso_ret20", "breadth_above_ema20", "breadth_rel20_positive",
]

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["close"]
    prev = c.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean() / c

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["close"].astype(float)
    v = x["volume"].fillna(0).astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    x["ret1"] = c.pct_change(1)
    x["ret5"] = c.pct_change(5)
    x["ret20"] = c.pct_change(20)
    x["ret63"] = c.pct_change(63)
    x["rsi14"] = rsi(c)
    x["above_ema20"] = (c > e20).astype(float)
    x["ema20_gt_ema50"] = (e20 > e50).astype(float)
    x["ema50_gt_ema200"] = (e50 > e200).astype(float)
    x["macd_delta_pct"] = (macd - sig) / c
    x["atr_pct"] = atr_pct(x)
    x["rv20"] = x["ret1"].rolling(20).std()
    x["volume_ratio20"] = v / v.rolling(20).mean().replace(0, np.nan)
    x["adv20"] = (c * v).rolling(20).mean()
    x["range_pct"] = (x["high"] - x["low"]) / c
    x["gap_pct"] = x["open"] / c.shift(1) - 1
    x["close_open_pct"] = c / x["open"] - 1
    hi = x["high"].rolling(252, min_periods=126).max()
    lo = x["low"].rolling(252, min_periods=126).min()
    x["pos_52w"] = (c - lo) / (hi - lo).replace(0, np.nan)
    return x

def download_all() -> dict[str, pd.DataFrame]:
    tickers = sorted(set(UNIVERSE + BENCHMARKS))
    raw = yf.download(
        tickers,
        start="2017-01-01",
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for s in tickers:
        try:
            d = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            d = d.rename(columns=str.lower)
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d = d.dropna(subset=["close"])
            if "volume" not in d:
                d["volume"] = 0.0
            out[s] = d[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception:
            out[s] = pd.DataFrame()
    return out

def weekly_features(d: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame({
        "open": d["open"].resample("W-FRI").first(),
        "high": d["high"].resample("W-FRI").max(),
        "low": d["low"].resample("W-FRI").min(),
        "close": d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna(subset=["close"])
    wc = w["close"]
    e10 = wc.ewm(span=10, adjust=False).mean()
    e20 = wc.ewm(span=20, adjust=False).mean()
    z = pd.DataFrame(index=w.index)
    z["weekly_ret1"] = wc.pct_change(1)
    z["weekly_ret4"] = wc.pct_change(4)
    z["weekly_above_ema10"] = (wc > e10).astype(float)
    z["weekly_ema10_gt_ema20"] = (e10 > e20).astype(float)
    return z.reindex(daily_index, method="ffill")

def market_features(frames: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    m = pd.DataFrame(index=index)
    spy = feature_frame(frames["SPY"]).reindex(index)
    qqq = feature_frame(frames["QQQ"]).reindex(index)
    iwm = feature_frame(frames["IWM"]).reindex(index)
    m["spy_ret5"] = spy["ret5"]
    m["spy_ret20"] = spy["ret20"]
    m["spy_ret63"] = spy["ret63"]
    m["spy_above_ema50"] = (spy["close"] > spy["close"].ewm(span=50, adjust=False).mean()).astype(float)
    m["spy_ema50_gt_ema200"] = (
        spy["close"].ewm(span=50, adjust=False).mean() > spy["close"].ewm(span=200, adjust=False).mean()
    ).astype(float)
    m["qqq_rel20_spy"] = qqq["ret20"] - spy["ret20"]
    m["iwm_rel20_spy"] = iwm["ret20"] - spy["ret20"]
    vix = frames["^VIX"]["close"].reindex(index).ffill()
    m["vix_level"] = vix
    m["vix_z20"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std().replace(0, np.nan)
    m["vix_ret5"] = vix.pct_change(5)
    for sym, name in [("TLT", "tlt_ret20"), ("GLD", "gld_ret20"), ("USO", "uso_ret20")]:
        s = frames[sym]["close"].reindex(index).ffill()
        m[name] = s.pct_change(20)
    return m

def daily_direction_score(row: pd.Series, spy_row: pd.Series, weekly_row: pd.Series) -> float:
    s = 0.0
    s += 10 if row["above_ema20"] > 0.5 else -10
    s += 10 if row["ema20_gt_ema50"] > 0.5 else -10
    if pd.notna(row["rsi14"]):
        s += 6 if row["rsi14"] >= 55 else -6 if row["rsi14"] <= 45 else 0
    if pd.notna(row["macd_delta_pct"]):
        s += 7 if row["macd_delta_pct"] >= 0 else -7
    if pd.notna(row["ret20"]) and pd.notna(spy_row.get("ret20")):
        s += 8 if row["ret20"] >= spy_row["ret20"] else -8
    if pd.notna(weekly_row.get("weekly_above_ema10")):
        s += 7 if weekly_row["weekly_above_ema10"] > 0.5 else -7
    if pd.notna(weekly_row.get("weekly_ema10_gt_ema20")):
        s += 6 if weekly_row["weekly_ema10_gt_ema20"] > 0.5 else -6
    if pd.notna(row["ret1"]):
        s += max(-10.0, min(10.0, float(row["ret1"]) * 100 * 3))
    return float(s)

def candidate_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = feature_frame(frames["SPY"])
    index = spy.index
    market = market_features(frames, index)

    feats: dict[str, pd.DataFrame] = {}
    weeks: dict[str, pd.DataFrame] = {}
    sectors: dict[str, pd.DataFrame] = {}
    for etf in SECTOR_GROUPS:
        if not frames.get(etf, pd.DataFrame()).empty:
            sectors[etf] = feature_frame(frames[etf]).reindex(index)
    for s in UNIVERSE:
        d = frames.get(s, pd.DataFrame())
        if d.empty or len(d) < 260:
            continue
        f = feature_frame(d).reindex(index)
        w = weekly_features(d, index)
        sec = sectors.get(SECTOR_OF[s])
        f["rel20_spy"] = f["ret20"] - spy["ret20"]
        f["rel63_spy"] = f["ret63"] - spy["ret63"]
        if sec is not None:
            f["sector_rel20"] = f["ret20"] - sec["ret20"]
            f["sector_rel63"] = f["ret63"] - sec["ret63"]
        else:
            f["sector_rel20"] = np.nan
            f["sector_rel63"] = np.nan
        feats[s] = f
        weeks[s] = w

    breadth = pd.DataFrame(index=index)
    breadth["breadth_above_ema20"] = pd.concat(
        [f["above_ema20"].rename(s) for s, f in feats.items()], axis=1
    ).mean(axis=1)
    breadth["breadth_rel20_positive"] = pd.concat(
        [(f["rel20_spy"] > 0).astype(float).rename(s) for s, f in feats.items()], axis=1
    ).mean(axis=1)
    market = market.join(breadth)

    rows = []
    eligible_dates = index[(index >= pd.Timestamp("2019-01-01")) & (index <= pd.Timestamp("2026-08-14"))]
    for dt in eligible_dates:
        pre = []
        spy_row = spy.loc[dt]
        for s, f in feats.items():
            try:
                r = f.loc[dt]
                w = weeks[s].loc[dt]
            except Exception:
                continue
            if pd.isna(r["close"]) or r["close"] < MIN_PRICE or pd.isna(r["adv20"]) or r["adv20"] < MIN_ADV20:
                continue
            score = daily_direction_score(r, spy_row, w)
            activity = abs(float(r["ret1"])) if pd.notna(r["ret1"]) else 0.0
            pre.append((s, score, activity))
        if len(pre) < 20:
            continue
        bull = sorted(pre, key=lambda z: z[1], reverse=True)[:3]
        bear = sorted(pre, key=lambda z: z[1])[:3]
        used = {x[0] for x in bull + bear}
        wildcard = next((x for x in sorted(pre, key=lambda z: z[2], reverse=True) if x[0] not in used), None)
        selected = [(x, "UP_PREFILTER", i + 1) for i, x in enumerate(bull)]
        selected += [(x, "DOWN_PREFILTER", i + 1) for i, x in enumerate(bear)]
        if wildcard:
            selected.append((wildcard, "WILDCARD", 1))

        for (s, score, activity), side, rank in selected:
            f = feats[s]
            r = f.loc[dt]
            w = weeks[s].loc[dt]
            record = {
                "date": dt,
                "symbol": s,
                "sector_etf": SECTOR_OF[s],
                "candidate_side": side,
                "candidate_side_code": 1.0 if side == "UP_PREFILTER" else -1.0 if side == "DOWN_PREFILTER" else 0.0,
                "candidate_direction_score": score,
                "candidate_activity": activity,
                "candidate_rank_abs": float(rank),
                "adv20_log": math.log(max(float(r["adv20"]), 1.0)),
            }
            for c in ["ret1","ret5","ret20","ret63","rsi14","above_ema20","ema20_gt_ema50","ema50_gt_ema200",
                      "macd_delta_pct","atr_pct","rv20","volume_ratio20","range_pct","gap_pct","close_open_pct",
                      "pos_52w","rel20_spy","rel63_spy","sector_rel20","sector_rel63"]:
                record[c] = r.get(c, np.nan)
            for c in ["weekly_ret1","weekly_ret4","weekly_above_ema10","weekly_ema10_gt_ema20"]:
                record[c] = w.get(c, np.nan)
            for c in market.columns:
                record[c] = market.at[dt, c] if dt in market.index else np.nan

            stock_close = f["close"]
            spy_close = spy["close"]
            for h in HORIZONS:
                try:
                    pos = index.get_loc(dt)
                    future_dt = index[pos + h]
                    stock_future = stock_close.loc[future_dt]
                    spy_future = spy_close.loc[future_dt]
                    if pd.isna(stock_future) or pd.isna(spy_future):
                        raise ValueError
                    sr = float(stock_future / r["close"] - 1)
                    br = float(spy_future / spy_row["close"] - 1)
                    excess = sr - br
                    vol = float(r["atr_pct"]) if pd.notna(r["atr_pct"]) else 0.02
                    threshold = max(0.004 * math.sqrt(h), 0.30 * vol * math.sqrt(h))
                    threshold = min(threshold, 0.08)
                    record[f"excess_{h}"] = excess
                    record[f"threshold_{h}"] = threshold
                    record[f"y_up_{h}"] = float(excess > threshold)
                    record[f"y_down_{h}"] = float(excess < -threshold)
                except Exception:
                    record[f"excess_{h}"] = np.nan
                    record[f"threshold_{h}"] = np.nan
                    record[f"y_up_{h}"] = np.nan
                    record[f"y_down_{h}"] = np.nan
            rows.append(record)
    return pd.DataFrame(rows)

def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    return (center - adj) / den

def calibration_fit(raw_p: np.ndarray, y: np.ndarray):
    x = np.log(np.clip(raw_p, 1e-5, 1-1e-5) / np.clip(1-raw_p, 1e-5, 1)).reshape(-1, 1)
    cal = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    cal.fit(x, y)
    return cal

def apply_cal(cal, raw_p):
    x = np.log(np.clip(raw_p, 1e-5, 1-1e-5) / np.clip(1-raw_p, 1e-5, 1)).reshape(-1, 1)
    return cal.predict_proba(x)[:, 1]

def models():
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=1200, class_weight="balanced", random_state=42)),
        ]),
        "hgb2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=2, learning_rate=0.04, max_iter=220,
                l2_regularization=2.0, random_state=42)),
        ]),
        "hgb4": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.035, max_iter=220,
                l2_regularization=3.0, random_state=42)),
        ]),
        "extra": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=350, max_depth=8, min_samples_leaf=25,
                class_weight="balanced", n_jobs=-1, random_state=42)),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        ]),
    }

def pick_threshold(df: pd.DataFrame, p: np.ndarray, target: str, side: str):
    base = float(df[target].mean())
    best = None
    for th in np.arange(0.50, 0.76, 0.025):
        mask = p >= th
        n = int(mask.sum())
        if n < 60:
            continue
        y = df.loc[mask, target].astype(int).to_numpy()
        obs = float(y.mean())
        meanp = float(p[mask].mean())
        lower = wilson_lower(int(y.sum()), n)
        gap = abs(meanp - obs)
        if obs < 0.58 or lower <= base + 0.02 or gap > 0.08:
            continue
        score = lower + 0.05 * min(n / 500.0, 1.0)
        cand = {"side": side, "threshold": round(float(th), 3), "n": n, "observed": obs,
                "mean_pred": meanp, "wilson_lower": lower, "calibration_gap": gap, "base_rate": base,
                "score": score}
        if best is None or score > best["score"]:
            best = cand
    return best

def precision_by_date(df: pd.DataFrame, p: np.ndarray, target: str, k: int):
    z = df[["date", target]].copy()
    z["p"] = p
    picks = []
    for _, g in z.groupby("date"):
        picks.append(g.nlargest(min(k, len(g)), "p"))
    if not picks:
        return {"n": 0, "precision": None, "wilson_lower": None}
    q = pd.concat(picks)
    y = q[target].astype(int).to_numpy()
    n = len(y)
    return {"n": n, "precision": float(y.mean()), "wilson_lower": wilson_lower(int(y.sum()), n)}

def evaluate_side(data: pd.DataFrame, h: int, side: str):
    target = f"y_{side}_{h}"
    side_name = "UP_PREFILTER" if side == "up" else "DOWN_PREFILTER"
    d = data[data["candidate_side"] == side_name].copy()
    d = d[d[target].notna()].copy()
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = d[target].astype(int)
    dates = pd.to_datetime(d["date"])

    train = dates < pd.Timestamp("2024-01-01")
    cal = (dates >= pd.Timestamp("2024-01-01")) & (dates < pd.Timestamp("2025-01-01"))
    select = (dates >= pd.Timestamp("2025-01-01")) & (dates < FINAL_START)
    final = dates >= FINAL_START
    if min(train.sum(), cal.sum(), select.sum(), final.sum()) < 250:
        raise RuntimeError(f"insufficient {side} h{h} samples")

    trials = []
    chosen = None
    for name, model in models().items():
        model.fit(X[train], y[train])
        raw_cal = model.predict_proba(X[cal])[:, 1]
        calibrator = calibration_fit(raw_cal, y[cal].to_numpy())
        p_sel = apply_cal(calibrator, model.predict_proba(X[select])[:, 1])
        sel_df = d.loc[select].copy()
        tail = pick_threshold(sel_df, p_sel, target, side)
        cal_ll = float(log_loss(y[cal], np.clip(apply_cal(calibrator, raw_cal), 1e-6, 1-1e-6)))
        trials.append({"model": name, "cal_logloss": cal_ll, "tail": tail})
        score = tail["score"] if tail else -1.0
        fallback = -cal_ll / 100.0
        objective = score if tail else fallback
        if chosen is None or objective > chosen["objective"]:
            chosen = {"name": name, "model": model, "calibrator": calibrator,
                      "tail": tail, "objective": objective, "cal_logloss": cal_ll}

    model = chosen["model"]
    calibrator = chosen["calibrator"]
    p_final = apply_cal(calibrator, model.predict_proba(X[final])[:, 1])
    yf = y[final].to_numpy()
    base_train = float(y[train].mean())
    base_final = float(y[final].mean())
    brier = float(brier_score_loss(yf, p_final))
    base_brier = float(brier_score_loss(yf, np.full(len(yf), base_train)))
    ll = float(log_loss(yf, np.clip(p_final, 1e-6, 1-1e-6)))
    base_ll = float(log_loss(yf, np.full(len(yf), np.clip(base_train, 1e-6, 1-1e-6))))
    auc = float(roc_auc_score(yf, p_final)) if len(np.unique(yf)) > 1 else None

    tail = chosen["tail"]
    final_tail = {"accepted": False, "n": 0}
    if tail:
        mask = p_final >= tail["threshold"]
        n = int(mask.sum())
        if n:
            yy = yf[mask]
            obs = float(yy.mean())
            meanp = float(p_final[mask].mean())
            lower = wilson_lower(int(yy.sum()), n)
            gap = abs(meanp - obs)
            p1 = precision_by_date(d.loc[final].copy(), p_final, target, 1)
            p3 = precision_by_date(d.loc[final].copy(), p_final, target, 3)
            accepted = (
                n >= 50 and obs >= 0.58 and lower > base_final + 0.02 and gap <= 0.08
                and p1["wilson_lower"] is not None and p1["wilson_lower"] > base_final
                and brier < base_brier and ll < base_ll
            )
            final_tail = {
                "accepted": bool(accepted), "n": n, "observed": obs, "mean_pred": meanp,
                "wilson_lower": lower, "calibration_gap": gap, "base_rate": base_final,
                "precision_at_1": p1, "precision_at_3": p3,
            }

    return {
        "model": chosen["name"],
        "target": f"P(meaningful benchmark-relative {side}) among historical {side_name} candidates",
        "train_n": int(train.sum()), "calibration_n": int(cal.sum()),
        "selection_2025_n": int(select.sum()), "final_2026_n": int(final.sum()),
        "base_rate_train": base_train, "base_rate_final": base_final,
        "brier": brier, "base_brier": base_brier,
        "log_loss": ll, "base_log_loss": base_ll, "roc_auc": auc,
        "selection_threshold": tail, "final_tail": final_tail,
        "accepted_for_display": bool(final_tail.get("accepted")),
        "model_trials": trials,
    }, {"model": model, "calibrator": calibrator, "threshold": tail, "features": FEATURES}

def main():
    frames = download_all()
    usable = [s for s in UNIVERSE if not frames.get(s, pd.DataFrame()).empty]
    print("usable universe", len(usable), "of", len(UNIVERSE))
    data = candidate_dataset(frames)
    print("candidate rows", len(data), "dates", data["date"].min(), data["date"].max())
    results = {}
    saved = {}
    accepted = []
    for h in HORIZONS:
        results[str(h)] = {}
        saved[str(h)] = {}
        for side in ["up", "down"]:
            metrics, obj = evaluate_side(data, h, side)
            results[str(h)][side] = metrics
            saved[str(h)][side] = obj
            if metrics["accepted_for_display"]:
                accepted.append({"horizon": h, "side": side})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_CONDITIONED_VALIDATED" if accepted else "NO_CANDIDATE_PROBABILITY_PASSED",
        "objective": "A few highly liquid candidates with calibrated probabilities conditional on passing the historical Phase1 shortlist funnel.",
        "universe_requested": len(UNIVERSE),
        "universe_usable": len(usable),
        "candidate_rows": int(len(data)),
        "date_start": str(data["date"].min().date()),
        "date_end": str(data["date"].max().date()),
        "funnel": "price >= $5, 20D average dollar volume >= $100M, then top 3 daily-direction bullish + bottom 3 bearish + one activity wildcard",
        "label": "Meaningful stock return relative to SPY, threshold=max(0.4%*sqrt(h), 0.30*ATR%*sqrt(h)), capped at 8%",
        "validation": "pre-2024 train; 2024 calibration; 2025 model/threshold selection; 2026+ untouched final test",
        "accepted": accepted,
        "metrics": results,
        "limitations": [
            "current-survivor universe; delisted names are not reconstructed",
            "daily/weekly candidate-conditioning model; live 15m/4h remains a separate refinement layer",
            "current sector membership mapping is used historically",
            "probabilities remain withheld unless the untouched final test passes",
        ],
    }
    OUT_VALIDATION.write_text(json.dumps(payload, separators=(",", ":")))
    joblib.dump({"generated_at": payload["generated_at"], "models": saved, "accepted": accepted,
                 "universe": UNIVERSE, "sector_of": SECTOR_OF, "features": FEATURES}, OUT_MODEL)
    print(json.dumps({
        "status": payload["status"], "universe_usable": len(usable),
        "candidate_rows": len(data), "accepted": accepted,
    }, indent=2))

if __name__ == "__main__":
    main()
