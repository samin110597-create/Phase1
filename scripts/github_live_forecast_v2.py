from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.live_probability_inference import calibrated_probabilities
from forecast.signal_gate import evaluate_signal
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
    market_open = bool(payload.get('market', {}).get('market_open'))
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
        row['signal_engine'] = evaluate_signal(row, market_open)
        if probs.get('status') != 'MODEL_NOT_TRAINED':
            any_model = True
        validated_horizons.update(str(h) for h in probs.get('validated_horizons', []))
        current_display_horizons.update(str(h) for h in probs.get('current_display_horizons', []))

    valid = [r for r in payload.get('rows', []) if not r.get('error') and not r.get('live_stale')]

    def p_of(r, h='5', side='up'):
        p = r.get('probability_model', {}).get('horizons', {}).get(str(h), {})
        val = p.get('p_up') if side == 'up' else p.get('p_down')
        return float(val) if val is not None else None

    def accepted_prob(r, h='5', side='up'):
        p = r.get('probability_model', {}).get('horizons', {}).get(str(h), {})
        needed = 'UP' if side == 'up' else 'DOWN'
        if not p.get('accepted_for_display') or p.get('display_side') != needed:
            return None
        return p.get('p_up') if side == 'up' else p.get('p_down')

    research_up = [(p_of(r, '5', 'up'), r) for r in valid]
    research_down = [(p_of(r, '5', 'down'), r) for r in valid]
    research_up = [x for x in research_up if x[0] is not None]
    research_down = [x for x in research_down if x[0] is not None]

    strict_up = [(accepted_prob(r, '5', 'up'), r) for r in valid]
    strict_down = [(accepted_prob(r, '5', 'down'), r) for r in valid]
    strict_up = [x for x in strict_up if x[0] is not None]
    strict_down = [x for x in strict_down if x[0] is not None]

    payload['research_top_upside'] = [r['symbol'] for _, r in sorted(research_up, key=lambda x: x[0], reverse=True)[:3]]
    payload['research_top_downside'] = [r['symbol'] for _, r in sorted(research_down, key=lambda x: x[0], reverse=True)[:3]]
    payload['validated_top_upside'] = [r['symbol'] for _, r in sorted(strict_up, key=lambda x: x[0], reverse=True)[:3]]
    payload['validated_top_downside'] = [r['symbol'] for _, r in sorted(strict_down, key=lambda x: x[0], reverse=True)[:3]]

    priority = {
        'STRICT_VALIDATED_SIGNAL_CANDIDATE': 4,
        'EVIDENCE_ALIGNED_SIGNAL_CANDIDATE': 3,
        'WATCHLIST': 2,
        'NO_SIGNAL': 1,
    }
    signal_rows = []
    for r in valid:
        s = r.get('signal_engine', {}).get('best', {})
        status = r.get('signal_engine', {}).get('status', 'NO_SIGNAL')
        signal_rows.append((priority.get(status, 0), float(s.get('probability') or 0), r))
    signal_rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    qualified = [r for rank, _, r in signal_rows if rank >= 3]
    watch = [r for rank, _, r in signal_rows if rank == 2]
    payload['qualified_signal_candidates'] = [r['symbol'] for r in qualified[:3]]
    payload['signal_watchlist'] = [r['symbol'] for r in watch[:5]]
    payload['signal_state'] = 'QUALIFIED SIGNAL CANDIDATE' if qualified else 'WATCHLIST ONLY' if watch else 'NO QUALIFIED SIGNAL'
    payload['signal_policy'] = 'A signal candidate requires probability + 15m/4h/Day/Week alignment + fresh data + direction/horizon backtest edge. Otherwise the engine abstains.'

    # Backward-compatible top lists are research rankings; their status is explicit below.
    payload['top_upside'] = payload['research_top_upside']
    payload['top_downside'] = payload['research_top_downside']

    if validated_horizons:
        payload['ranking_basis'] = 'research ranking shown; strict-validated candidates are separately identified when a frozen threshold passes'
    elif any_model:
        payload['ranking_basis'] = 'research probabilities shown; selective signal gate abstains unless probability, MTF structure, freshness and historical edge agree'
    else:
        payload['ranking_basis'] = 'probability model not trained'

    payload['accepted_probability_horizons'] = sorted(validated_horizons, key=int) if validated_horizons else []
    payload['current_display_horizons'] = sorted(current_display_horizons, key=int) if current_display_horizons else []
    if validated_horizons:
        payload['status'] = 'RESEARCH PROBABILITIES AVAILABLE — STRICT VALIDATION PASSED FOR: ' + ','.join(sorted(validated_horizons, key=int))
    elif any_model:
        payload['status'] = 'RESEARCH PROBABILITIES AVAILABLE — SELECTIVE SIGNAL GATE ACTIVE'
    else:
        payload['status'] = 'PROBABILITY MODEL NOT TRAINED YET'

    payload['truth_policy'] = {
        'research_estimates_visible': True,
        'research_estimates_are_not_claimed_as_validated': True,
        'selective_signal_gate_active': True,
        'signal_engine_can_abstain': True,
        'strict_validated_candidates_only_when_holdout_gate_passes': True,
        'failed_backtests_are_retained_and_displayed': True,
    }

    OUT.write_text(json.dumps(payload, separators=(',', ':')))
    print(json.dumps({
        'generated_at': payload.get('generated_at'),
        'status': payload.get('status'),
        'signal_state': payload.get('signal_state'),
        'qualified_signal_candidates': payload.get('qualified_signal_candidates'),
        'signal_watchlist': payload.get('signal_watchlist'),
        'accepted_probability_horizons': payload.get('accepted_probability_horizons'),
        'research_top_upside': payload.get('research_top_upside'),
        'research_top_downside': payload.get('research_top_downside'),
        'validated_top_upside': payload.get('validated_top_upside'),
        'validated_top_downside': payload.get('validated_top_downside'),
    }, indent=2))


if __name__ == '__main__':
    main()
