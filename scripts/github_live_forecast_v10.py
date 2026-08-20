from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast_v7 as stable
import scripts.github_live_forecast_v9 as hourly
from forecast.live_intraday_v3_inference import available, bundle_version, activation_status

OUT = Path('docs/data/live_forecast.json')


def evidence_ready() -> bool:
    return bool(available() and bundle_version() == 'hourly-meta-v7')


def main():
    if evidence_ready():
        hourly.main()
        return

    # Keep the last working probability + numeric-target engine active whenever a
    # staged replacement has not earned promotion through its own evidence gate.
    stable.main()
    payload = json.loads(OUT.read_text())
    gate = activation_status()
    payload['production_model_policy'] = {
        'status': 'STABLE_MODEL_ACTIVE',
        'active_model': (payload.get('rows') or [{}])[0].get('probability_model', {}).get('model_version'),
        'staged_hourly_model': bundle_version(),
        'staged_model_accepted': False,
        'staged_model_status': 'NOT_PROMOTED',
        'staged_model_reason': gate.get('reason'),
        'policy': 'A staged model may replace production only after its own historical evidence gate passes; otherwise the working probability/target engine remains active.'
    }
    payload['status'] = 'MODEL FORECASTS ACTIVE — EVIDENCE-GATED PRODUCTION'
    payload['hourly_meta_v7_layer'] = {
        'status': 'STAGED_NOT_PROMOTED',
        'model_version': bundle_version(),
        'reason': gate.get('reason')
    }
    OUT.write_text(json.dumps(payload, separators=(',', ':')))


if __name__ == '__main__':
    main()
