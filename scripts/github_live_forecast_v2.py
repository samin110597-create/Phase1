from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.live_probability_inference import calibrated_probabilities
import scripts.github_live_forecast as base

OUT = Path('docs/data/live_forecast.json')


def main():
    base.main()
    payload = json.loads(OUT.read_text())

    try:
        spy15_df = base.twelve_15m('SPY')
        spy15 = base.frame_features(spy15_df, '15m')
    except Exception:
        spy15 = None

    now_ny = datetime.now(base.NY)
    minute = now_ny.hour * 60 + now_ny.minute
    validated_horizons = set()
    current_display_horizons = set()
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
        validated_horizons.update(str(h) for h in probs.get('validated_horizons', []))
        current_display_horizons.update(str(h) for h in probs.get('current_display_horizons', []))

    valid = [r for r in payload.get('rows', []) if not r.get('error') and not r.get('live_stale')]

    def accepted_prob(r, h='5', side='up'):
        p = r.get('probability_model',{}).get('horizons',{}).get(h,{})
        needed = 'UP' if side == 'up' else 'DOWN'
        if not p.get('accepted_for_display') or p.get('display_side') != needed:
            return None
        return p.get('p_up') if side == 'up' else p.get('p_down')

    ups = [(accepted_prob(r,'5','up'), r) for r in valid]
    dns = [(accepted_prob(r,'5','down'), r) for r in valid]
    ups = [x for x in ups if x[0] is not None]
    dns = [x for x in dns if x[0] is not None]
    payload['top_upside'] = [r['symbol'] for _,r in sorted(ups,key=lambda x:x[0],reverse=True)[:3]]
    payload['top_downside'] = [r['symbol'] for _,r in sorted(dns,key=lambda x:x[0],reverse=True)[:3]]

    if '5' not in validated_horizons:
        payload['ranking_basis'] = '5-session model failed strict validation — candidates withheld'
    elif not ups and not dns:
        payload['ranking_basis'] = '5-session model validated, but no current stock clears its frozen high-probability threshold'
    else:
        payload['ranking_basis'] = 'strictly validated 5-session calibrated probability with 2025-frozen threshold and 2026 untouched holdout'

    payload['accepted_probability_horizons'] = sorted(validated_horizons,key=int) if validated_horizons else []
    payload['current_display_horizons'] = sorted(current_display_horizons,key=int) if current_display_horizons else []
    if validated_horizons:
        payload['status'] = 'STRICT CALIBRATED PROBABILITY MODEL — VALIDATED HORIZONS: ' + ','.join(sorted(validated_horizons,key=int))
    elif any_model:
        payload['status'] = 'NO HORIZON PASSED STRICT PROBABILITY VALIDATION'
    else:
        payload['status'] = 'PROBABILITY MODEL NOT TRAINED YET'

    OUT.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({
        'generated_at':payload.get('generated_at'),'status':payload.get('status'),
        'accepted_probability_horizons':payload.get('accepted_probability_horizons'),
        'current_display_horizons':payload.get('current_display_horizons'),
        'ranking_basis':payload.get('ranking_basis'),'top_upside':payload.get('top_upside'),'top_downside':payload.get('top_downside')
    },indent=2))


if __name__ == '__main__':
    main()
