from __future__ import annotations

from pathlib import Path
import forecast.train_hourly_selective_v8 as trainer

# Never overwrite the production model during the first evidence run.
trainer.LIVE_MODEL = Path('forecast/data/hourly_selective_v8_promotable.joblib')

if __name__ == '__main__':
    trainer.main()
