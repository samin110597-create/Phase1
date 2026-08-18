from __future__ import annotations

import scripts.github_live_forecast_v9 as v9
from forecast.live_intraday_v3_inference import available, bundle_version


def evidence_ready() -> bool:
    # Production activation is allowed only when the frozen hourly bundle
    # passes the independent evidence gate in live_intraday_v3_inference.
    return bool(available() and bundle_version() == 'hourly-meta-v7')


v9.hourly_ready = evidence_ready

if __name__ == '__main__':
    v9.main()
