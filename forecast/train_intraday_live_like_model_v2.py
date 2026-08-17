from __future__ import annotations

import forecast.train_intraday_live_like_model as base

# Candidate-ranking fields are used to form the historical research pool but
# are not available from the expensive live 15m layer for all 40 names. The
# decision-time price proxy exists only to align outcome labels with the live
# logger. None of these fields may enter the probability model.
EXCLUDE_FROM_MODEL = {
    'candidate_score',
    'candidate_rank_pct',
    'activity_rank_pct',
    'rvol_rank_pct',
    'cross_section_rel_rank',
    'decision_price_proxy',
}

_original_feature_columns = base.feature_columns


def live_reproducible_features(df):
    return [c for c in _original_feature_columns(df) if c not in EXCLUDE_FROM_MODEL]


base.feature_columns = live_reproducible_features

if __name__ == '__main__':
    base.main()
