from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast as base
import scripts.github_live_forecast_v4 as engine

OUT = Path('docs/data/live_forecast.json')
FROZEN_SLOTS = [(10,0),(12,0),(14,0),(15,45)]
SLOT_WINDOW_MINUTES = 7


def frozen_slot(now_ny: datetime):
    minute=now_ny.hour*60+now_ny.minute
    for hh,mm in FROZEN_SLOTS:
        t=hh*60+mm
        if 0 <= minute-t < SLOT_WINDOW_MINUTES:
            return f'{hh:02d}:{mm:02d}'
    return None


def main():
    engine.main()
    payload=json.loads(OUT.read_text())
    if payload.get('engine') != 'Phase1 GitHub-Only Live Forecast Engine V3':
        OUT.write_text(json.dumps(payload,separators=(',',':')))
        return
    try:
        spyq=base.finnhub_quote('SPY')
        spy_price=float(spyq['price'])
        spy_age=spyq.get('age_seconds')
    except Exception:
        spy_price=None; spy_age=None
    payload['benchmark_snapshot']={'symbol':'SPY','price':spy_price,'age_seconds':spy_age,'provider':'Finnhub' if spy_price else None}
    now_ny=datetime.now(base.NY); slot=frozen_slot(now_ny)
    payload['frozen_decision_slot']=slot
    payload['frozen_decision_windows_et']=['10:00','12:00','14:00','15:45']
    market_open=bool(payload.get('market',{}).get('market_open'))
    qualified=[]
    for row in payload.get('rows',[]):
        if row.get('error'): continue
        row['benchmark_snapshot_price']=spy_price
        day=row.get('day') or {}; atr_pct=day.get('atr_pct')
        atr_frac=(float(atr_pct)/100.0) if atr_pct is not None else 0.02
        for h,hp in (row.get('probability_model') or {}).get('horizons',{}).items():
            try: hh=int(h)
            except Exception: continue
            hp['target_move_threshold']=round(max(0.003,0.35*max(0.005,atr_frac)*math.sqrt(hh)),6)
            hp['target_definition']='meaningful stock excess return versus SPY from the same intraday snapshot'
        se=row.get('signal_engine') or {}
        if market_open and slot is None and se.get('status')=='FROZEN_FORWARD_SIGNAL_CANDIDATE':
            se['status']='OUTSIDE_FROZEN_DECISION_WINDOW'
            if se.get('best'): se['best']['status']='OUTSIDE_FROZEN_DECISION_WINDOW'
        if se.get('status')=='FROZEN_FORWARD_SIGNAL_CANDIDATE':
            qualified.append(row['symbol'])
        row['signal_engine']=se
    payload['qualified_signal_candidates']=qualified[:3]
    payload['signal_state']='FROZEN FORWARD SIGNAL CANDIDATE' if qualified else 'NO QUALIFIED SIGNAL'
    payload['truth_policy']=payload.get('truth_policy',{})
    payload['truth_policy']['forward_target_matches_training_target']=bool(spy_price)
    payload['truth_policy']['signal_candidates_only_inside_frozen_decision_windows']=True
    OUT.write_text(json.dumps(payload,separators=(',',':')))

if __name__=='__main__': main()
