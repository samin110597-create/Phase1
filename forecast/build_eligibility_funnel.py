from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

MTF=Path('docs/data/mtf_features.json')
LIVE=Path('docs/data/intraday.json')
OUT=Path('docs/data/forecast_funnel.json')


def n(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except Exception:return default


def main():
    mtf=json.loads(MTF.read_text()) if MTF.exists() else {'rows':[]}
    live=json.loads(LIVE.read_text()) if LIVE.exists() else {'quotes':[]}
    lm={x.get('symbol'):x for x in live.get('quotes',[]) if x.get('symbol')}
    passed=[]; rejected=[]
    for r in mtf.get('rows',[]):
        s=r.get('symbol');q=lm.get(s) or {}
        coverage=n(r.get('feature_coverage_pct'))
        dv=n(r.get('avg_dollar_volume'))
        fresh=int(q.get('providers_used') or 0)
        agree=n(q.get('agreement_pct'))
        spread=n(q.get('provider_spread_pct'),999)
        reasons=[]
        if coverage<75:reasons.append('multi-timeframe coverage <75%')
        if dv<50_000_000:reasons.append('20-day average dollar volume <$50M')
        if fresh<2:reasons.append('fewer than 2 fresh quote sources')
        if q and agree<80:reasons.append('quote agreement <80%')
        if q and spread>0.75:reasons.append('provider spread >0.75%')
        row={'symbol':s,'feature_coverage_pct':coverage,'avg_dollar_volume':r.get('avg_dollar_volume'),'fresh_quote_sources':fresh,'quote_agreement_pct':q.get('agreement_pct'),'provider_spread_pct':q.get('provider_spread_pct'),'eligible':not reasons,'reasons':reasons}
        if reasons:rejected.append(row)
        else:passed.append(row)
    passed.sort(key=lambda x:(x['feature_coverage_pct'],x['fresh_quote_sources'],n(x['avg_dollar_volume'])),reverse=True)
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'FORECAST ELIGIBILITY FUNNEL — NOT A DIRECTIONAL MODEL','rules':'Requires >=75% 15m/4h/day/week feature coverage, >=$50M average dollar volume, >=2 fresh quote sources, >=80% quote agreement, and <=0.75% provider spread where available.','eligible_count':len(passed),'eligible_symbols':passed,'rejected_count':len(rejected),'rejected_symbols':rejected}
    OUT.write_text(json.dumps(payload,separators=(',',':')))
    print('forecast eligibility:',len(passed),'eligible /',len(passed)+len(rejected),'evaluated')

if __name__=='__main__':main()
