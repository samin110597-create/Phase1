from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.github_live_forecast_v6 as engine

OUT = Path('docs/data/live_forecast.json')


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def build_price_forecast(row: dict, horizon: int, hp: dict) -> dict | None:
    price = _f((row.get('quote') or {}).get('price'))
    atr_pct = _f((row.get('day') or {}).get('atr_pct'))
    p_up = _f(hp.get('p_up'))
    p_down = _f(hp.get('p_down'))
    if not price or price <= 0 or atr_pct is None or p_up is None or p_down is None:
        return None

    # Daily ATR is already stored as percent of price (e.g. 2.3 = 2.3%).
    atr_frac = max(0.001, atr_pct / 100.0)
    hvol = atr_frac * math.sqrt(max(1, horizon))

    # Probability-weighted central forecast. This is intentionally transparent:
    # directional probability edge (P(up)-P(down)) × horizon-scaled ATR.
    edge = p_up - p_down
    central_move = edge * hvol
    central_target = price * (1.0 + central_move)

    # One horizon-scaled ATR band around the current price.
    range_low = max(0.01, price * (1.0 - hvol))
    range_high = price * (1.0 + hvol)

    # Meaningful-move threshold used as a practical minimum move reference.
    meaningful_move = max(0.003, 0.35 * max(0.005, atr_frac) * math.sqrt(max(1, horizon)))
    meaningful_up = price * (1.0 + meaningful_move)
    meaningful_down = price * (1.0 - meaningful_move)

    direction = 'UP' if p_up > p_down else 'DOWN' if p_down > p_up else 'NEUTRAL'
    direction_probability = max(p_up, p_down)

    return {
        'horizon_sessions': horizon,
        'current_price': round(price, 4),
        'forecast_direction': direction,
        'direction_probability': round(direction_probability, 4),
        'central_expected_move_pct': round(central_move * 100.0, 3),
        'central_target_price': round(central_target, 2),
        'projected_range_low': round(range_low, 2),
        'projected_range_high': round(range_high, 2),
        'meaningful_move_pct': round(meaningful_move * 100.0, 3),
        'meaningful_up_price': round(meaningful_up, 2),
        'meaningful_down_price': round(meaningful_down, 2),
        'daily_atr_pct': round(atr_pct, 3),
        'method': 'central target = current price × [1 + (P(up)-P(down)) × daily ATR% × sqrt(horizon)]; range = ±1 horizon-scaled ATR',
        'status': 'VALIDATED_MODEL_FORECAST' if hp.get('accepted_for_display') else 'MODEL_FORECAST_VALIDATION_PENDING',
    }


def main():
    engine.main()
    payload = json.loads(OUT.read_text())

    for row in payload.get('rows', []):
        if row.get('error'):
            continue
        forecasts = {}
        horizons = ((row.get('probability_model') or {}).get('horizons') or {})
        for h, hp in horizons.items():
            try:
                hi = int(h)
            except Exception:
                continue
            pf = build_price_forecast(row, hi, hp)
            if pf:
                forecasts[str(hi)] = pf
                hp['price_forecast'] = pf
        row['price_forecasts'] = forecasts

    payload['price_target_engine'] = {
        'status': 'ACTIVE',
        'version': 'price-target-v1',
        'outputs': ['central_target_price', 'central_expected_move_pct', 'projected_range_low', 'projected_range_high', 'meaningful_up_price', 'meaningful_down_price'],
        'method': 'probability-weighted directional edge combined with current daily ATR and square-root-of-time horizon scaling',
        'truth_note': 'These are model-implied quantitative targets, not guaranteed future prices. Validation status remains separate from the forecast itself.'
    }

    # User-facing semantics: show the forecast, while retaining the evidence state separately.
    payload['forecast_display_policy'] = {
        'show_model_forecasts_even_when_validation_pending': True,
        'label_unvalidated_output': 'MODEL FORECAST · VALIDATION PENDING',
        'label_validated_output': 'VALIDATED MODEL FORECAST',
        'never_relabel_validation_pending_as_validated': True,
    }

    OUT.write_text(json.dumps(payload, separators=(',', ':')))


if __name__ == '__main__':
    main()
