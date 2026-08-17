from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from forecast import train_candidate_conditioned_probability as base

OUT_MODEL = Path("forecast/data/ranked_candidate_probability.joblib")
OUT_VALIDATION = Path("docs/data/ranked_candidate_validation.json")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_VALIDATION.parent.mkdir(parents=True, exist_ok=True)

POOL_K = 10
HORIZONS = [1, 5, 10]
FINAL_START = pd.Timestamp("2026-01-01")

RANK_FEATURES = [
    "xs_direction_rank", "xs_activity_rank", "xs_ret20_rank", "xs_ret63_rank",
    "xs_rel20_rank", "xs_sector_rel20_rank", "xs_volume_rank", "xs_rsi_rank", "xs_pos52_rank",
    "regime_code", "direction_x_breadth", "sector_rel_x_vix",
]
FEATURES = base.FEATURES + RANK_FEATURES

def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")

def regime_code(m: pd.Series) -> float:
    trend = bool(m.get("spy_above_ema50", 0) > 0.5 and m.get("spy_ema50_gt_ema200", 0) > 0.5)
    high_vol = bool(pd.notna(m.get("vix_level")) and m.get("vix_level") >= 20)
    if trend and not high_vol:
        return 0.0
    if trend and high_vol:
        return 1.0
    if not trend and not high_vol:
        return 2.0
    return 3.0

def ranked_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = base.feature_frame(frames["SPY"])
    index = spy.index
    market = base.market_features(frames, index)

    feats: dict[str, pd.DataFrame] = {}
    weeks: dict[str, pd.DataFrame] = {}
    sectors: dict[str, pd.DataFrame] = {}
    for etf in base.SECTOR_GROUPS:
        d = frames.get(etf, pd.DataFrame())
        if not d.empty:
            sectors[etf] = base.feature_frame(d).reindex(index)

    for s in base.UNIVERSE:
        d = frames.get(s, pd.DataFrame())
        if d.empty or len(d) < 260:
            continue
        f = base.feature_frame(d).reindex(index)
        w = base.weekly_features(d, index)
        sec = sectors.get(base.SECTOR_OF[s])
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
    dates = index[(index >= pd.Timestamp("2019-01-01")) & (index <= pd.Timestamp("2026-08-14"))]
    for dt in dates:
        snap = []
        spy_row = spy.loc[dt]
        for s, f in feats.items():
            r = f.loc[dt]
            w = weeks[s].loc[dt]
            if pd.isna(r["close"]) or r["close"] < base.MIN_PRICE or pd.isna(r["adv20"]) or r["adv20"] < base.MIN_ADV20:
                continue
            score = base.daily_direction_score(r, spy_row, w)
            activity = abs(float(r["ret1"])) if pd.notna(r["ret1"]) else 0.0
            snap.append({
                "symbol": s,
                "direction_score": score,
                "activity": activity,
                "ret20": r.get("ret20"),
                "ret63": r.get("ret63"),
                "rel20": r.get("rel20_spy"),
                "sector_rel20": r.get("sector_rel20"),
                "volume_ratio20": r.get("volume_ratio20"),
                "rsi14": r.get("rsi14"),
                "pos52": r.get("pos_52w"),
            })
        if len(snap) < 30:
            continue
        cross = pd.DataFrame(snap).set_index("symbol")
        cross["xs_direction_rank"] = pct_rank(cross["direction_score"])
        cross["xs_activity_rank"] = pct_rank(cross["activity"])
        cross["xs_ret20_rank"] = pct_rank(cross["ret20"])
        cross["xs_ret63_rank"] = pct_rank(cross["ret63"])
        cross["xs_rel20_rank"] = pct_rank(cross["rel20"])
        cross["xs_sector_rel20_rank"] = pct_rank(cross["sector_rel20"])
        cross["xs_volume_rank"] = pct_rank(cross["volume_ratio20"])
        cross["xs_rsi_rank"] = pct_rank(cross["rsi14"])
        cross["xs_pos52_rank"] = pct_rank(cross["pos52"])

        bull_names = cross.nlargest(POOL_K, "direction_score").index.tolist()
        bear_names = cross.nsmallest(POOL_K, "direction_score").index.tolist()
        selected = [(s, "UP_POOL", i + 1) for i, s in enumerate(bull_names)]
        selected += [(s, "DOWN_POOL", i + 1) for i, s in enumerate(bear_names)]

        m = market.loc[dt]
        reg = regime_code(m)
        for s, side, rank in selected:
            f = feats[s]
            r = f.loc[dt]
            w = weeks[s].loc[dt]
            xs = cross.loc[s]
            record = {
                "date": dt,
                "symbol": s,
                "sector_etf": base.SECTOR_OF[s],
                "candidate_side": side,
                "candidate_side_code": 1.0 if side == "UP_POOL" else -1.0,
                "candidate_direction_score": float(xs["direction_score"]),
                "candidate_activity": float(xs["activity"]),
                "candidate_rank_abs": float(rank),
                "adv20_log": math.log(max(float(r["adv20"]), 1.0)),
                "regime_code": reg,
            }
            for c in [
                "ret1","ret5","ret20","ret63","rsi14","above_ema20","ema20_gt_ema50","ema50_gt_ema200",
                "macd_delta_pct","atr_pct","rv20","volume_ratio20","range_pct","gap_pct","close_open_pct",
                "pos_52w","rel20_spy","rel63_spy","sector_rel20","sector_rel63",
            ]:
                record[c] = r.get(c, np.nan)
            for c in ["weekly_ret1","weekly_ret4","weekly_above_ema10","weekly_ema10_gt_ema20"]:
                record[c] = w.get(c, np.nan)
            for c in market.columns:
                record[c] = m.get(c, np.nan)
            for c in [
                "xs_direction_rank","xs_activity_rank","xs_ret20_rank","xs_ret63_rank","xs_rel20_rank",
                "xs_sector_rel20_rank","xs_volume_rank","xs_rsi_rank","xs_pos52_rank",
            ]:
                record[c] = xs.get(c, np.nan)
            breadth_now = float(m.get("breadth_above_ema20")) if pd.notna(m.get("breadth_above_ema20")) else 0.5
            vix_now = float(m.get("vix_level")) if pd.notna(m.get("vix_level")) else 20.0
            sector_rel = float(r.get("sector_rel20")) if pd.notna(r.get("sector_rel20")) else 0.0
            record["direction_x_breadth"] = float(xs["direction_score"]) * (breadth_now - 0.5)
            record["sector_rel_x_vix"] = sector_rel * max(vix_now - 15.0, 0.0)

            pos = index.get_loc(dt)
            for h in HORIZONS:
                try:
                    future_dt = index[pos + h]
                    stock_future = f.at[future_dt, "close"]
                    spy_future = spy.at[future_dt, "close"]
                    if pd.isna(stock_future) or pd.isna(spy_future):
                        raise ValueError
                    stock_ret = float(stock_future / r["close"] - 1)
                    spy_ret = float(spy_future / spy_row["close"] - 1)
                    excess = stock_ret - spy_ret
                    vol = float(r["atr_pct"]) if pd.notna(r["atr_pct"]) else 0.02
                    th = max(0.004 * math.sqrt(h), 0.30 * vol * math.sqrt(h))
                    th = min(th, 0.08)
                    record[f"excess_{h}"] = excess
                    record[f"y_up_{h}"] = float(excess > th)
                    record[f"y_down_{h}"] = float(excess < -th)
                except Exception:
                    record[f"excess_{h}"] = np.nan
                    record[f"y_up_{h}"] = np.nan
                    record[f"y_down_{h}"] = np.nan
            rows.append(record)
    return pd.DataFrame(rows)

def daily_picks(df: pd.DataFrame, p: np.ndarray, target: str, threshold: float, k: int):
    z = df[["date", target]].copy()
    z["p"] = p
    picks = []
    for _, g in z.groupby("date"):
        q = g[g["p"] >= threshold].nlargest(k, "p")
        if not q.empty:
            picks.append(q)
    if not picks:
        return {"n": 0, "days": 0, "precision": None, "wilson_lower": None, "mean_probability": None}
    q = pd.concat(picks)
    y = q[target].astype(int).to_numpy()
    return {
        "n": int(len(q)),
        "days": int(q["date"].nunique()),
        "precision": float(y.mean()),
        "wilson_lower": base.wilson_lower(int(y.sum()), len(y)),
        "mean_probability": float(q["p"].mean()),
    }

def choose_threshold(df: pd.DataFrame, p: np.ndarray, target: str):
    base_rate = float(df[target].mean())
    best = None
    for th in np.arange(0.35, 0.76, 0.025):
        p1 = daily_picks(df, p, target, float(th), 1)
        p3 = daily_picks(df, p, target, float(th), 3)
        if p1["n"] < 70 or p1["precision"] is None:
            continue
        gap = abs(p1["mean_probability"] - p1["precision"])
        if p1["precision"] < 0.58 or p1["wilson_lower"] <= base_rate + 0.05 or gap > 0.10:
            continue
        score = p1["wilson_lower"] + 0.15 * min(p1["days"] / 200.0, 1.0)
        cand = {
            "threshold": round(float(th), 3),
            "base_rate": base_rate,
            "precision_at_1": p1,
            "precision_at_3": p3,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = cand
    return best

def evaluate(data: pd.DataFrame, h: int, side: str):
    target = f"y_{side}_{h}"
    pool_side = "UP_POOL" if side == "up" else "DOWN_POOL"
    d = data[(data["candidate_side"] == pool_side) & data[target].notna()].copy()
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = d[target].astype(int)
    dates = pd.to_datetime(d["date"])

    train = dates < pd.Timestamp("2024-01-01")
    cal = (dates >= pd.Timestamp("2024-01-01")) & (dates < pd.Timestamp("2025-01-01"))
    select = (dates >= pd.Timestamp("2025-01-01")) & (dates < FINAL_START)
    final = dates >= FINAL_START
    if min(train.sum(), cal.sum(), select.sum(), final.sum()) < 500:
        raise RuntimeError(f"insufficient ranked pool samples h={h} side={side}")

    trials = []
    chosen = None
    for name, model in base.models().items():
        model.fit(X[train], y[train])
        raw_cal = model.predict_proba(X[cal])[:, 1]
        calibrator = base.calibration_fit(raw_cal, y[cal].to_numpy())
        cal_p = base.apply_cal(calibrator, raw_cal)
        p_sel = base.apply_cal(calibrator, model.predict_proba(X[select])[:, 1])
        sel_df = d.loc[select].copy()
        threshold = choose_threshold(sel_df, p_sel, target)
        cal_ll = float(log_loss(y[cal], np.clip(cal_p, 1e-6, 1 - 1e-6)))
        trials.append({"model": name, "cal_logloss": cal_ll, "selection": threshold})
        objective = threshold["score"] if threshold else -1.0 - cal_ll / 100.0
        if chosen is None or objective > chosen["objective"]:
            chosen = {
                "name": name, "model": model, "calibrator": calibrator,
                "selection": threshold, "objective": objective, "cal_logloss": cal_ll,
            }

    p_final = base.apply_cal(chosen["calibrator"], chosen["model"].predict_proba(X[final])[:, 1])
    yf = y[final].to_numpy()
    base_train = float(y[train].mean())
    base_final = float(y[final].mean())
    brier = float(brier_score_loss(yf, p_final))
    base_brier = float(brier_score_loss(yf, np.full(len(yf), base_train)))
    ll = float(log_loss(yf, np.clip(p_final, 1e-6, 1 - 1e-6)))
    base_ll = float(log_loss(yf, np.full(len(yf), np.clip(base_train, 1e-6, 1 - 1e-6))))
    auc = float(roc_auc_score(yf, p_final)) if len(np.unique(yf)) > 1 else None

    final_test = {"accepted": False, "n": 0}
    sel = chosen["selection"]
    if sel:
        df_final = d.loc[final].copy()
        p1 = daily_picks(df_final, p_final, target, sel["threshold"], 1)
        p3 = daily_picks(df_final, p_final, target, sel["threshold"], 3)
        if p1["n"]:
            gap = abs(p1["mean_probability"] - p1["precision"])
            accepted = (
                p1["n"] >= 60
                and p1["precision"] >= 0.60
                and p1["wilson_lower"] > base_final + 0.05
                and gap <= 0.10
                and p3["precision"] is not None
                and p3["precision"] >= 0.52
                and p3["wilson_lower"] > base_final + 0.03
                and brier < base_brier
                and ll < base_ll
            )
            final_test = {
                "accepted": bool(accepted),
                "threshold": sel["threshold"],
                "base_rate": base_final,
                "precision_at_1": p1,
                "precision_at_3": p3,
                "calibration_gap_at_1": gap,
            }

    return {
        "model": chosen["name"],
        "target": f"P(meaningful SPY-relative {side}) within daily top-{POOL_K} {pool_side}",
        "train_n": int(train.sum()), "calibration_n": int(cal.sum()),
        "selection_2025_n": int(select.sum()), "final_2026_n": int(final.sum()),
        "base_rate_train": base_train, "base_rate_final": base_final,
        "brier": brier, "base_brier": base_brier,
        "log_loss": ll, "base_log_loss": base_ll, "roc_auc": auc,
        "selection": sel, "final_test": final_test,
        "accepted_for_display": bool(final_test.get("accepted")),
        "model_trials": trials,
    }, {
        "model": chosen["model"],
        "calibrator": chosen["calibrator"],
        "selection": sel,
        "features": FEATURES,
    }

def main():
    frames = base.download_all()
    usable = [s for s in base.UNIVERSE if not frames.get(s, pd.DataFrame()).empty]
    data = ranked_dataset(frames)
    print("usable", len(usable), "pool rows", len(data))
    results = {}
    saved = {}
    accepted = []
    for h in HORIZONS:
        results[str(h)] = {}
        saved[str(h)] = {}
        for side in ["up", "down"]:
            metrics, obj = evaluate(data, h, side)
            results[str(h)][side] = metrics
            saved[str(h)][side] = obj
            if metrics["accepted_for_display"]:
                accepted.append({"horizon": h, "side": side})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RANKED_CANDIDATE_PROBABILITY_VALIDATED" if accepted else "NO_RANKED_PROBABILITY_PASSED",
        "objective": "Let the model choose 1-3 names from a wider top-10 bullish / top-10 bearish liquid candidate pool.",
        "universe_requested": len(base.UNIVERSE),
        "universe_usable": len(usable),
        "pool_k_each_side": POOL_K,
        "candidate_rows": int(len(data)),
        "date_start": str(data["date"].min().date()),
        "date_end": str(data["date"].max().date()),
        "features": "daily + weekly + SPY/QQQ/IWM/VIX/TLT/GLD/USO + sector-relative + market breadth + cross-sectional ranks",
        "label": "meaningful stock excess return versus SPY using ATR/horizon-adjusted threshold",
        "validation": "pre-2024 train; 2024 calibration; 2025 model/threshold/Precision@K selection; 2026+ untouched final test",
        "acceptance": "final Precision@1 >=60%, Precision@3 >=52%, Wilson bounds above final base rate, calibration gap <=10%, and Brier/log-loss beat base",
        "accepted": accepted,
        "metrics": results,
        "limitations": [
            "current-survivor universe; delisted stocks are not reconstructed",
            "current sector membership is used historically",
            "this is the daily/weekly ranking layer; live 15m/4h validation remains a separate final gate",
            "no probability is published unless the untouched 2026 gate passes",
        ],
    }
    OUT_VALIDATION.write_text(json.dumps(payload, separators=(",", ":")))
    joblib.dump({
        "generated_at": payload["generated_at"], "accepted": accepted,
        "models": saved, "features": FEATURES, "pool_k": POOL_K,
        "universe": base.UNIVERSE, "sector_of": base.SECTOR_OF,
    }, OUT_MODEL)
    print(json.dumps({
        "status": payload["status"], "candidate_rows": len(data), "accepted": accepted
    }, indent=2))

if __name__ == "__main__":
    main()
