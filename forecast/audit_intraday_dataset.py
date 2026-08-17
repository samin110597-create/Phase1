from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA=Path('forecast/data/intraday_live_like_v2.csv.gz')
OUT=Path('docs/data/intraday_dataset_audit.json')
EXPECTED_SLOTS={'10:00','12:00','14:00','15:45'}


def main():
    if not DATA.exists(): raise RuntimeError('V3 dataset missing')
    df=pd.read_csv(DATA,compression='gzip',parse_dates=['snapshot_dt'])
    failures=[]; checks={}

    keys=['snapshot_dt','symbol']
    dup=int(df.duplicated(keys).sum())
    checks['duplicate_snapshot_symbol_rows']=dup
    if dup: failures.append(f'{dup} duplicate snapshot/symbol rows')

    slots=set(df['slot'].dropna().astype(str).unique())
    checks['slots_found']=sorted(slots)
    unexpected=sorted(slots-EXPECTED_SLOTS)
    if unexpected: failures.append('unexpected decision slots: '+','.join(unexpected))

    if 'decision_price_proxy' not in df:
        failures.append('decision_price_proxy missing after label correction')
        checks['decision_price_proxy_valid_rate']=0.0
    else:
        valid=(pd.to_numeric(df['decision_price_proxy'],errors='coerce')>0)
        rate=float(valid.mean())
        checks['decision_price_proxy_valid_rate']=rate
        if rate<0.97: failures.append(f'decision price valid rate too low: {rate:.3%}')

    label_stats={}
    for h in (1,5,10):
        up=f'label_up_{h}'; dn=f'label_down_{h}'; ex=f'fwd_{h}_excess'; th=f'move_threshold_{h}'
        required=[up,dn,ex,th]
        missing=[c for c in required if c not in df]
        if missing:
            failures.append(f'h{h} missing columns: '+','.join(missing)); continue
        z=df[df[up].notna() & df[dn].notna() & df[ex].notna() & df[th].notna()].copy()
        both=int(((z[up]>0.5)&(z[dn]>0.5)).sum())
        wrong_up=int(((z[up]>0.5)&~(z[ex]>z[th])).sum())
        wrong_dn=int(((z[dn]>0.5)&~(z[ex]<-z[th])).sum())
        contradictory=int(((z[up]<=0.5)&(z[dn]<=0.5)&((z[ex]>z[th])|(z[ex]<-z[th]))).sum())
        labeled_rate=float(len(z)/len(df)) if len(df) else 0.0
        label_stats[str(h)]={
            'labeled_rows':int(len(z)),'labeled_rate':labeled_rate,
            'up_rate':float(z[up].mean()) if len(z) else None,'down_rate':float(z[dn].mean()) if len(z) else None,
            'both_up_and_down':both,'wrong_up_labels':wrong_up,'wrong_down_labels':wrong_dn,'missed_directional_labels':contradictory,
            'threshold_min':float(z[th].min()) if len(z) else None,'threshold_median':float(z[th].median()) if len(z) else None,'threshold_max':float(z[th].max()) if len(z) else None,
        }
        if both or wrong_up or wrong_dn or contradictory:
            failures.append(f'h{h} label consistency failure: both={both}, wrong_up={wrong_up}, wrong_down={wrong_dn}, missed={contradictory}')
        min_rate={1:0.95,5:0.94,10:0.92}[h]
        if labeled_rate<min_rate: failures.append(f'h{h} labeled rate too low: {labeled_rate:.3%}')
        if len(z) and (z[th]<=0).any(): failures.append(f'h{h} has non-positive movement thresholds')

    checks['label_stats']=label_stats
    checks['rows']=int(len(df)); checks['symbols']=int(df['symbol'].nunique())
    checks['first_snapshot']=str(df['snapshot_dt'].min()); checks['last_snapshot']=str(df['snapshot_dt'].max())
    checks['split_like_rows_remaining']=int((pd.to_numeric(df.get('split_like_gap',0),errors='coerce').fillna(0)>0.5).sum()) if 'split_like_gap' in df else None
    if checks['split_like_rows_remaining'] not in (None,0): failures.append('split-like gap rows remain in training dataset')

    # Feature-safety audit: labels/outcomes and the decision price proxy are never allowed
    # to masquerade as model inputs. The trainer has its own exclusion list; this audit
    # records the dangerous columns explicitly for review.
    checks['forbidden_model_input_patterns']=['fwd_*','spy_fwd_*','label_*','move_threshold_*','decision_price_proxy','candidate_rank fields']
    checks['chronological_period_counts']={
        'pre_2024':int((df['snapshot_dt']<pd.Timestamp('2024-01-01')).sum()),
        '2024':int(((df['snapshot_dt']>=pd.Timestamp('2024-01-01'))&(df['snapshot_dt']<pd.Timestamp('2025-01-01'))).sum()),
        '2025':int(((df['snapshot_dt']>=pd.Timestamp('2025-01-01'))&(df['snapshot_dt']<pd.Timestamp('2026-01-01'))).sum()),
        '2026_development':int((df['snapshot_dt']>=pd.Timestamp('2026-01-01')).sum()),
    }
    for name,n in checks['chronological_period_counts'].items():
        if n<500: failures.append(f'chronological block {name} is too small: {n}')

    result={
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'status':'PASS' if not failures else 'FAIL',
        'checks':checks,
        'failures':failures,
        'truth_note':'This audit checks structural/label integrity. Passing it does not establish predictive accuracy; it only permits model training.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps(result,indent=2))
    if failures: raise RuntimeError('V3 dataset audit failed: '+'; '.join(failures[:8]))

if __name__=='__main__': main()
