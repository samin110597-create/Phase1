from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

import forecast.train_rank_consensus_v9 as v9
import forecast.sec_event_features as sec

STAGE=Path('forecast/data/event_rank_consensus_v10.joblib')
VAL=Path('docs/data/event_rank_consensus_v10_validation.json')
STATUS=Path('docs/data/event_rank_consensus_v10_status.json')
for p in (STAGE,VAL,STATUS):p.parent.mkdir(parents=True,exist_ok=True)

_ORIG_BUILD=v9.v8.build_dataset


def event_build_dataset():
    data,coverage,missing=_ORIG_BUILD()
    augmented=sec.add_sec_features(data,refresh=False)
    return augmented,coverage,missing


def main():
    v9.v8.build_dataset=event_build_dataset
    v9.STAGE=STAGE; v9.VAL=VAL; v9.STATUS=STATUS
    v9.main()

    if STAGE.exists():
        b=joblib.load(STAGE); b['version']='event-rank-consensus-v10'; b['sec_event_features']=list(sec.EVENT_FEATURES); b['architecture_note']='V9 rank-consensus architecture plus point-in-time SEC 8-K/10-Q/10-K filing-event features'; joblib.dump(b,STAGE,compress=3)
    if VAL.exists():
        x=json.loads(VAL.read_text()); x['model_version']='event-rank-consensus-v10'; x['status']=x.get('status','').replace('V9','V10 SEC-EVENT'); x['sec_event_features']=list(sec.EVENT_FEATURES); x['architecture']=x.get('architecture','')+' + point-in-time SEC filing-event recency/count features'; x['event_source']='SEC EDGAR submissions history; exact acceptance timestamp when available, otherwise next-day availability fallback'; VAL.write_text(json.dumps(x,separators=(',',':')))
    if STATUS.exists():
        x=json.loads(STATUS.read_text()); x['model_version']='event-rank-consensus-v10'; x['sec_event_features']=list(sec.EVENT_FEATURES); STATUS.write_text(json.dumps(x,separators=(',',':')))
    print(json.dumps({'status':'V10_SEC_EVENT_TRAINING_COMPLETE','stage_model':STAGE.exists(),'validation':VAL.exists(),'event_features':sec.EVENT_FEATURES},indent=2))

if __name__=='__main__':main()
