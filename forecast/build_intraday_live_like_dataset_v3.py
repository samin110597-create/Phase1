from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import forecast.build_intraday_live_like_dataset as base


def fast_make_snapshots(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    z=base.intraday_feature_frame(bars)
    rows=[]
    prev_close=None
    for date,g in z.groupby('date',sort=True):
        g=g.sort_values('datetime')
        first_open=float(g.iloc[0]['open'])
        split_like=bool(prev_close is not None and prev_close>0 and abs(first_open/prev_close-1)>0.35)
        for slot in base.SLOTS:
            decision=pd.Timestamp(f'{date.date()} {slot}')
            cutoff=decision-pd.Timedelta(minutes=15)
            elig=g[g['datetime']<=cutoff]
            if elig.empty:
                continue
            r=elig.iloc[-1].copy()
            r['snapshot_dt']=decision
            r['slot']=slot
            r['symbol']=symbol
            r['split_like_gap']=float(split_like)
            rows.append(r)
        prev_close=float(g.iloc[-1]['close'])
    if not rows:
        return pd.DataFrame()
    out=pd.DataFrame(rows)
    keep=['snapshot_dt','slot','symbol','close','split_like_gap','m15_ret1','m15_ret3','m15_ret10','m15_rsi14','m15_above_ema20','m15_ema20_gt_ema50','m15_macd_delta_pct','m15_atr_pct','m15_vol_ratio20','m15_vwap20_dist','h4_ret','h4_ret3','h4_range_pct','h4_vol_ratio','session_ret','session_range_pos','session_vwap_dist','session_rvol','volume_accel','opening_range_pos']
    return out[keep]


base.make_snapshots=fast_make_snapshots

if __name__=='__main__':
    base.main()
