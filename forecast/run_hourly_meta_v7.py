from __future__ import annotations

import pandas as pd

from forecast import train_hourly_meta_model_v7 as trainer


def fixed_asof_prev(snaps: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty or feat.empty:
        return snaps
    left = snaps[['snapshot_dt']].copy()
    left['date'] = pd.to_datetime(left['snapshot_dt'], errors='coerce').dt.normalize().astype('datetime64[ns]')
    idx_name = feat.index.name or 'index'
    right = feat.reset_index().rename(columns={idx_name: 'context_date'}).copy()
    right['context_date'] = pd.to_datetime(right['context_date'], errors='coerce').astype('datetime64[ns]')
    left = left.dropna(subset=['date']).sort_values('date')
    right = right.dropna(subset=['context_date']).sort_values('context_date')
    merged = pd.merge_asof(
        left,
        right,
        left_on='date',
        right_on='context_date',
        direction='backward',
        allow_exact_matches=False,
    )
    out = snaps.sort_values('snapshot_dt').copy()
    for c in feat.columns:
        out[c] = merged[c].to_numpy()
    return out


trainer.asof_prev = fixed_asof_prev

if __name__ == '__main__':
    trainer.main()
