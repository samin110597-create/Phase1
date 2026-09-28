# Phase1 — Q-State Challenger Research Lab

Phase1 is **not a production forecasting model** and no longer publishes an independent BUY/SELL, probability, price target, or stock-analysis view.

The only production model is **Q-State Unified** in `samin110597-create/stock-truth-v2`:

- Canonical terminal: https://samin110597-create.github.io/stock-truth-v2/quant/
- Canonical Deno API: https://stock-truth-v2.samin110597.deno.net
- Canonical model artifact: `stock-truth-v2/data/quant/model.json`

## Purpose

Phase1 is an offline challenger/validation lab. Existing V2–V10 research code and evidence can be retained for reproducibility, but it has **zero production weight**.

A Phase1 idea may influence production only when all of the following are true:

1. feature timing is reproducible at decision time and passes leakage checks;
2. validation is chronological with purging/embargo where required;
3. the challenger beats the current Q-State baseline on untouched/OOS data;
4. probability calibration is not worse;
5. improvement is stable across more than one market regime and is not driven by one ticker/sector;
6. ablation shows the new feature/model adds incremental value;
7. the winning method is incorporated into the **single Q-State Unified artifact** and revalidated there.

Phase1 never becomes a second live model and does not vote with Q-State.

## Shared data source

New Phase1 experiments should use `scripts/qstate_client.py` to retrieve the same Deno-validated market and research context used by Q-State. Do not create a separate live provider-routing stack unless it is a temporary test fixture.

The Deno service performs request-time data routing, freshness checks, cross-provider validation, fundamentals retrieval, and macro context. API credentials remain outside browser code.

## Reproducibility

The currently serialized research bundles were created under scikit-learn 1.9.0. Runtime dependencies are pinned to that version to prevent silent estimator incompatibility. New challenger artifacts must record their Python/scikit-learn versions and training timestamp.

## Existing research assets

Historical scripts, validation files, and forward logs remain available as evidence. They are research artifacts only and must not be described as the current production forecast.

The GitHub Pages URL for this repository redirects to Q-State Unified so there is one user-facing forecast/research/analysis model.
