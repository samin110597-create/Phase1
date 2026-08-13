# Phase1 Forecast V1

Objective: produce a very small, selective set of research candidates with calibrated directional probabilities for 1, 5, and 10 trading sessions, while abstaining when evidence is weak.

Core inputs: validated current quote, 15-minute structure, 4-hour structure, daily context, weekly context, benchmark/sector relative strength, liquidity/data quality, market regime, and timestamped corporate-event flags.

Validation: rolling walk-forward training, separate calibration periods, untouched holdout testing, probability reliability checks, Brier/log-loss reporting, and prospective forecast logging.

Primary evaluation: Precision@1, Precision@3, and Precision@5 rather than broad-universe accuracy. The system should be allowed to return NO QUALIFIED SETUP.

Legacy Confluence remains a diagnostic feature only and must not be presented as probability.
