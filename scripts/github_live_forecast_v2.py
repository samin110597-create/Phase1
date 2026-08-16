from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from forecast.live_probability_inference import calibrated_probabilities
import scripts.github_live_forecast as base

OUT = Path('docs/data/live_forecast.json')


def main():
    # Build the existing GitHub-only near-live snapshot first.
    base.main()
    payload = json.loads(OUT.read_text())

    try:
        spy15_df = base.twelve_15m('SPY')
        spy15 = base.frame_features(spy15_df, '15m')
    except Exception:
        spy15 = None

    now_ny = datetime.now(base.NY)
    minute = now_ny.hour * 60 + now_ny.minute
    accepted_horizons = set()
    any_model = False

    for row in payload.get('rows', []):
        if row.get('error'):
            continue
        probs = calibrated_probabilities(
            row.get('m15'), row.get('h4'), row.get('day'), row.get('week'),
            row.get('quote'), spy15, minute
        )
        row['probability_model'] = probs
        if probs.get('status') != 'MODEL_NOT_TRAINED':
            any_model = True
        for h, p in probs.get('horizons', {}).items():
            if p.get('accepted_for_display'):
                accepted_horizons.add(str(h))

    # Default ranking uses the 5-session calibrated probability when validated.
    valid = [r for r in payload.get('rows', []) if not r.get('error') and not r.get('live_stale')]
    def accepted_prob(r, h='5', side='up'):
        p = r.get('probability_model',{}).get('horizons',{}).get(h,{})
        if not p.get('accepted_for_display'):
            return None
        return p.get('p_up') if side == 'up' else p.get('p_down')

    if '5' in accepted_horizons:
        ups = [(accepted_prob(r,'5','up'), r) for r in valid]
        dns = [(accepted_prob(r,'5','down'), r) for r in valid]
        ups = [x for x in ups if x[0] is not None]
        dns = [x for x in dns if x[0] is not None]
        payload['top_upside'] = [r['symbol'] for _,r in sorted(ups,key=lambda x:x[0],reverse=True)[:3]]
        payload['top_downside'] = [r['symbol'] for _,r in sorted(dns,key=lambda x:x[0],reverse=True)[:3]]
        payload['ranking_basis'] = 'validated 5-session calibrated probability'
    else:
        payload['top_upside'] = []
        payload['top_downside'] = []
        payload['ranking_basis'] = 'NO VALIDATED 5-SESSION PROBABILITY — candidates withheld'

    payload['accepted_probability_horizons'] = sorted(accepted_horizons, key=int) if accepted_horizons else []
    if accepted_horizons:
        payload['status'] = 'CALIBRATED PROBABILITIES — VALIDATED V1 HORIZONS: ' + ','.join(sorted(accepted_horizons,key=int))
    elif any_model:
        payload['status'] = 'CALIBRATED MODEL EXISTS BUT FAILED DISPLAY VALIDATION — NO PROBABILITIES RANKED'
    else:
        payload['status'] = 'PROBABILITY MODEL NOT TRAINED YET'

    OUT.write_text(json.dumps(payload, separators=(',',':')))
    print(json.dumps({
        'generated_at': payload.get('generated_at'),
        'status': payload.get('status'),
        'accepted_probability_horizons': payload.get('accepted_probability_horizons'),
        'ranking_basis': payload.get('ranking_basis'),
        'top_upside': payload.get('top_upside'),
        'top_downside': payload.get('top_downside'),
    }, indent=2))


if __name__ == '__main__':
    main()
