from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import forecast.train_live_probability as trainer


def raw_daily_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(symbols, start='2019-01-01', auto_adjust=False, actions=False,
                      group_by='ticker', threads=True, progress=False)
    out = {}
    for symbol in symbols:
        try:
            d = raw[symbol].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            d = d.rename(columns=str.lower).dropna(subset=['close'])
            d.index = pd.to_datetime(d.index).tz_localize(None).astype('datetime64[ns]')
            out[symbol] = d[['open','high','low','close','volume']].copy()
        except Exception:
            out[symbol] = pd.DataFrame()
    return out


def candidate_models():
    return {
        'logistic': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(C=0.5, max_iter=1200, random_state=42)),
        ]),
        'hgb_depth2': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingClassifier(max_depth=2, learning_rate=0.04,
                max_iter=220, l2_regularization=2.0, random_state=42)),
        ]),
        'hgb_depth4': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingClassifier(max_depth=4, learning_rate=0.04,
                max_iter=220, l2_regularization=2.0, random_state=42)),
        ]),
        'extra_trees': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesClassifier(n_estimators=350, max_depth=9,
                min_samples_leaf=35, max_features='sqrt', n_jobs=-1, random_state=42)),
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=350, max_depth=9,
                min_samples_leaf=35, max_features='sqrt', n_jobs=-1, random_state=42)),
        ]),
    }


def fit_calibrator(model, Xc, yc):
    raw = np.clip(model.predict_proba(Xc)[:,1], 1e-5, 1-1e-5)
    cal = LogisticRegression(C=1e6, max_iter=500, random_state=42)
    cal.fit(trainer.safe_logit(raw), yc)
    return cal


def calibrated(cal, model, X):
    raw = np.clip(model.predict_proba(X)[:,1], 1e-5, 1-1e-5)
    return cal.predict_proba(trainer.safe_logit(raw))[:,1]


def selection_score(y, p):
    up, _ = trainer.choose_threshold(y, p, 'up')
    down, _ = trainer.choose_threshold(y, p, 'down')
    viable = [(x['wilson_lower'], x['n'], side, x) for side,x in [('up',up),('down',down)] if x]
    if not viable:
        return None, up, down
    best = max(viable, key=lambda z: (z[0], z[1]))
    return best, up, down


def precision_fit_horizon(data: pd.DataFrame, h: int):
    target = f'y_up_{h}'
    d = data.loc[data[target].notna()].copy().reset_index(drop=True)
    X = d[trainer.FEATURES].replace([np.inf,-np.inf], np.nan)
    y = d[target].astype(int)
    dates = pd.to_datetime(d['snapshot_date'])
    train = dates < pd.Timestamp('2024-01-01')
    calmask = (dates >= pd.Timestamp('2024-01-01')) & (dates < pd.Timestamp('2025-01-01'))
    select = (dates >= pd.Timestamp('2025-01-01')) & (dates < pd.Timestamp('2026-01-01'))
    final = dates >= pd.Timestamp('2026-01-01')
    Xtr,ytr = X[train],y[train]; Xc,yc = X[calmask],y[calmask]; Xs,ys = X[select],y[select]; Xf,yf_ = X[final],y[final]
    if min(len(Xtr),len(Xc),len(Xs),len(Xf)) < 1000:
        raise RuntimeError(f'horizon {h}: insufficient chronological samples')

    trials = []
    fitted = []
    for name, model in candidate_models().items():
        model.fit(Xtr, ytr)
        cal = fit_calibrator(model, Xc, yc)
        pc = calibrated(cal, model, Xc)
        ps = calibrated(cal, model, Xs)
        best_tail, up_sel, down_sel = selection_score(ys, ps)
        cal_loss = float(log_loss(yc, np.clip(pc,1e-6,1-1e-6)))
        trial = {
            'model':name, 'calibration_log_loss':round(cal_loss,5),
            'best_tail_side':best_tail[2] if best_tail else None,
            'best_tail_wilson_lower':round(best_tail[0],4) if best_tail else None,
            'best_tail_n':best_tail[1] if best_tail else 0,
            'up_threshold':up_sel['threshold'] if up_sel else None,
            'down_threshold':down_sel['threshold'] if down_sel else None,
        }
        print('horizon',h,'selection trial',trial)
        trials.append(trial)
        fitted.append((name,model,cal,cal_loss,best_tail,up_sel,down_sel))

    tail_models = [x for x in fitted if x[4] is not None]
    if tail_models:
        chosen = max(tail_models, key=lambda x: (x[4][0], x[4][1], -x[3]))
        selection_objective = 'highest 2025 qualified high-probability Wilson lower bound'
    else:
        chosen = min(fitted, key=lambda x: x[3])
        selection_objective = 'fallback: lowest 2024 calibration log loss; no 2025 tail qualified'

    chosen_name, model, cal, _, _, up_sel, down_sel = chosen
    pf = calibrated(cal, model, Xf)
    base = float(ytr.mean()); basef = np.full(len(yf_),base)
    brier=float(brier_score_loss(yf_,pf)); bb=float(brier_score_loss(yf_,basef))
    ll=float(log_loss(yf_,np.clip(pf,1e-6,1-1e-6))); bll=float(log_loss(yf_,np.clip(basef,1e-6,1-1e-6)))
    auc=float(roc_auc_score(yf_,pf)) if len(np.unique(yf_))>1 else None
    ece,bins=trainer.ece_and_bins(yf_,pf)
    overall_ok=bool(brier<bb and ll<bll and ece<=0.05)

    up_final=trainer.threshold_eval(yf_,pf,'up',up_sel['threshold']) if up_sel else {'accepted':False,'n':0}
    down_final=trainer.threshold_eval(yf_,pf,'down',down_sel['threshold']) if down_sel else {'accepted':False,'n':0}
    ff=d.loc[final,['datetime',target]].copy(); ff['p']=pf
    for side,selx,finx in [('up',up_sel,up_final),('down',down_sel,down_final)]:
        if not selx:
            continue
        finx['precision_at_1']=trainer.precision_at_k(ff,'p',target,1,side,selx['threshold'])
        finx['precision_at_3']=trainer.precision_at_k(ff,'p',target,3,side,selx['threshold'])
        finx['precision_at_5']=trainer.precision_at_k(ff,'p',target,5,side,selx['threshold'])
        p3=finx['precision_at_3']; base_side=finx.get('base_rate',0)
        finx['accepted']=bool(overall_ok and finx.get('accepted') and p3.get('picks',0)>=80 and
            p3.get('precision') is not None and p3['precision']>=0.58 and p3['wilson_lower_95']>base_side)

    accepted_sides=[side for side,x in [('up',up_final),('down',down_final)] if x.get('accepted')]
    metrics={
        'horizon_sessions':h,'model':chosen_name,'model_selection_objective':selection_objective,
        'model_trials':trials,'train_n':int(len(ytr)),'calibration_n':int(len(yc)),
        'selection_2025_n':int(len(ys)),'final_2026_n':int(len(yf_)),
        'final_start':'2026-01-01','final_end':str(dates[final].max().date()),
        'base_up_rate_train':round(base,4),'final_observed_up_rate':round(float(yf_.mean()),4),
        'brier':round(brier,5),'base_brier':round(bb,5),'log_loss':round(ll,5),'base_log_loss':round(bll,5),
        'roc_auc':round(auc,4) if auc is not None else None,'ece_10bin':round(ece,4),
        'overall_probability_quality_pass':overall_ok,'accepted_sides':accepted_sides,
        'accepted_for_display':bool(accepted_sides),'selection_thresholds':{'up':up_sel,'down':down_sel},
        'final_high_probability_tests':{'up':up_final,'down':down_final},'calibration_bins_final':bins,
    }
    bundle={'model':model,'calibrator':cal,'features':trainer.FEATURES,'metrics':metrics,
            'display_thresholds':{'up':up_sel['threshold'] if up_sel else None,'down':down_sel['threshold'] if down_sel else None}}
    return bundle,metrics


trainer.daily_history = raw_daily_history
trainer.fit_horizon = precision_fit_horizon

if __name__ == '__main__':
    trainer.main()
