from __future__ import annotations

import forecast.train_intraday_live_like_model as base

# These fields are useful to form the historical candidate pool, but the live
# intraday layer does not observe all 40 symbols with Twelve Data at once. Keep
# them out of the probability model so training and inference use the same
# reproducible feature set.
EXCLUDE_FROM_MODEL = {
    'candidate_score',
    'candidate_rank_pct',
    'activity_rank_pct',
    'rvol_rank_pct',
    'cross_section_rel_rank',
}

_original_feature_columns = base.feature_columns


def live_reproducible_features(df):
    return [c for c in _original_feature_columns(df) if c not in EXCLUDE_FROM_MODEL]


base.feature_columns = live_reproducible_features

if __name__ == '__main__':
    base.main()
