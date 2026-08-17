from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path('forecast/data/intraday_live_like_v2.csv.gz')
SUMMARY = Path('docs/data/intraday_training_summary.json')
CACHE = Path('forecast/cache/intraday15')
CACHE.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp(os.getenv('PHASE1_INTRADAY_START', '2020-01-01'))
MAX_SYMBOLS = int(os.getenv('PHASE1_INTRADAY_SYMBOLS', '40'))
TWELVE_KEY = os.getenv('TWELVE_DATA_API_KEY')
REQUESTS_PER_MINUTE = int(os.getenv('PHASE1_TWELVE_RPM', '7'))
TIMEOUT = 25
SLOTS = ['10:00', '12:00', '14:00', '15:45']

UNIVERSE = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AMD','AVGO',
    'NFLX','JPM','BAC','XOM','CVX','WMT','COST','HD','UNH','LLY','ORCL','CRM',
    'PLTR','MU','INTC','QCOM','AMAT','TSM','UBER','V','MA','GS','MS','CAT','GE',
    'DIS','KO','PEP','CSCO','ADBE'
][:MAX_SYMBOLS]

SECTOR_OF = {
    'AAPL':'XLK','MSFT':'XLK','NVDA':'XLK','AMD':'XLK','AVGO':'XLK','ORCL':'XLK','CRM':'XLK','ADBE':'XLK','CSCO':'XLK','INTC':'XLK','QCOM':'XLK','AMAT':'XLK','MU':'XLK','PLTR':'XLK',
    'META':'XLC','GOOGL':'XLC','GOOG':'XLC','NFLX':'XLC','DIS':'XLC',
    'AMZN':'XLY','TSLA':'XLY','HD':'XLY','UBER':'XLY',
    'JPM':'XLF','BAC':'XLF','V':'XLF','MA':'XLF','GS':'XLF','MS':'XLF',
    'UNH':'XLV','LLY':'XLV','XOM':'XLE','CVX':'XLE','CAT':'XLI','GE':'XLI',
    'WMT':'XLP','COST':'XLP','KO':'XLP','PEP':'XLP','TSM':'XLK'
}
BENCHMARKS = ['SPY','QQQ','IWM','^VIX','TLT','GLD','USO'] + sorted(set(SECTOR_OF.values()))


class RateLimiter:
    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self.used = 0
        self.window = time.monotonic()

    def wait(self):
        elapsed = time.monotonic() - self.window
        if elapsed >= 61:
            self.used = 0
            self.window = time.monotonic()
            elapsed = 0
        if self.used >= self.limit:
            sleep_for = max(0.0, 61.0 - elapsed)
            print(f'credit guard: sleeping {sleep_for:.1f}s')
            time.sleep(sleep_for)
            self.used = 0
            self.window = time.monotonic()
        self.used += 1


LIMITER = RateLimiter(REQUESTS_PER_MINUTE)


def request_json(url: str, retries: int = 4):
    last = None
    for attempt in range(retries):
        LIMITER.wait()
        try:
            req = Request(url, headers={'User-Agent': 'Phase1-Intraday-Training/2.0'})
            with urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode('utf-8'))
        except HTTPError as e:
            last = e
            if e.code == 429:
                delay = 65 * (attempt + 1)
                print(f'HTTP 429; backoff {delay}s')
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last if last else RuntimeError('request failed')


def parse_bars(vals) -> pd.DataFrame:
    x = pd.DataFrame(vals or [])
    if x.empty or 'datetime' not in x:
        return pd.DataFrame()
    x['datetime'] = pd.to_datetime(x['datetime'], errors='coerce')
    for c in ['open','high','low','close','volume']:
        x[c] = pd.to_numeric(x.get(c), errors='coerce')
    x = x.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime')
    return x[['datetime','open','high','low','close','volume']].drop_duplicates('datetime')


def fetch_15m(symbol: str) -> pd.DataFrame:
    if not TWELVE_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY missing')
    cache_path = CACHE / f'{symbol.replace("^", "IDX_")}.csv.gz'
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, compression='gzip', parse_dates=['datetime'])
            cached = parse_bars(cached.to_dict('records'))
        except Exception:
            cached = pd.DataFrame()

    target = START
    pieces = [cached] if not cached.empty else []
    end_date = None
    seen_oldest = None
    # If cache already reaches target, only refresh the newest 10 calendar days.
    if not cached.empty and cached['datetime'].min().normalize() <= target:
        end_date = None
        target = max(target, cached['datetime'].max().normalize() - pd.Timedelta(days=10))

    for page in range(16):
        params = {
            'symbol': symbol, 'interval': '15min', 'outputsize': 5000,
            'timezone': 'America/New_York', 'adjust': 'splits', 'apikey': TWELVE_KEY,
        }
        if end_date:
            params['end_date'] = end_date
        d = request_json('https://api.twelvedata.com/time_series?' + urlencode(params))
        vals = d.get('values') if isinstance(d, dict) else None
        if not isinstance(vals, list) or not vals:
            msg = d.get('message') if isinstance(d, dict) else 'no values'
            print(symbol, 'stopped:', msg)
            break
        x = parse_bars(vals)
        if x.empty:
            break
        pieces.append(x)
        oldest = x['datetime'].min()
        print(symbol, 'page', page + 1, 'rows', len(x), 'oldest', oldest)
        if oldest.normalize() <= target:
            break
        if seen_oldest is not None and oldest >= seen_oldest:
            break
        seen_oldest = oldest
        end_date = (oldest - pd.Timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%S')

    if not pieces:
        return pd.DataFrame()
    z = pd.concat(pieces, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
    z = z[z['datetime'] >= START]
    tm = z['datetime'].dt.strftime('%H:%M')
    z = z[(tm >= '09:30') & (tm <= '15:45')].reset_index(drop=True)
    z.to_csv(cache_path, index=False, compression='gzip')
    return z


def rsi(s: pd.Series, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def intraday_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    z = bars.copy().sort_values('datetime').reset_index(drop=True)
    c = z['close'].astype(float); v = z['volume'].fillna(0).astype(float)
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([(z['high']-z['low']), (z['high']-prev).abs(), (z['low']-prev).abs()], axis=1).max(axis=1)
    typical = (z['high'] + z['low'] + z['close']) / 3
    z['m15_ret1'] = c.pct_change(1); z['m15_ret3'] = c.pct_change(3); z['m15_ret10'] = c.pct_change(10)
    z['m15_rsi14'] = rsi(c); z['m15_above_ema20'] = (c > e20).astype(float); z['m15_ema20_gt_ema50'] = (e20 > e50).astype(float)
    z['m15_macd_delta_pct'] = (macd-sig)/c; z['m15_atr_pct'] = tr.rolling(14).mean()/c
    z['m15_vol_ratio20'] = v / v.rolling(20).mean().replace(0, np.nan)
    z['m15_vwap20_dist'] = c / ((typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)) - 1
    z['h4_ret'] = c.pct_change(16); z['h4_ret3'] = c.pct_change(48)
    z['h4_range_pct'] = (z['high'].rolling(16).max()-z['low'].rolling(16).min())/c
    z['h4_vol_ratio'] = v.rolling(16).mean()/v.rolling(80).mean().replace(0,np.nan)

    z['date'] = z['datetime'].dt.normalize()
    z['bar_no'] = z.groupby('date').cumcount()
    z['session_open'] = z.groupby('date')['open'].transform('first')
    z['session_high'] = z.groupby('date')['high'].cummax(); z['session_low'] = z.groupby('date')['low'].cummin()
    z['session_ret'] = c / z['session_open'] - 1
    z['session_range_pos'] = (c-z['session_low'])/(z['session_high']-z['session_low']).replace(0,np.nan)
    z['cum_pv'] = (typical*v).groupby(z['date']).cumsum(); z['cum_vol'] = v.groupby(z['date']).cumsum()
    z['session_vwap_dist'] = c/(z['cum_pv']/z['cum_vol'].replace(0,np.nan))-1
    z['same_slot_vol20'] = z.groupby('bar_no')['volume'].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    z['session_rvol'] = v / z['same_slot_vol20'].replace(0,np.nan)
    z['volume_accel'] = v / v.shift(1).replace(0,np.nan)
    first2 = z[z['bar_no'] < 2].groupby('date').agg(or_high=('high','max'), or_low=('low','min'))
    z = z.join(first2, on='date')
    z['opening_range_pos'] = (c-z['or_low'])/(z['or_high']-z['or_low']).replace(0,np.nan)
    return z


def make_snapshots(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    z = intraday_feature_frame(bars)
    rows = []
    for date, g in z.groupby('date', sort=True):
        g = g.sort_values('datetime')
        # split-like discontinuity guard: large raw overnight gaps can poison unadjusted intraday history.
        first_open = float(g.iloc[0]['open'])
        prev = z[z['date'] < date]
        prev_close = float(prev.iloc[-1]['close']) if not prev.empty else np.nan
        split_like = bool(pd.notna(prev_close) and prev_close > 0 and abs(first_open/prev_close - 1) > 0.35)
        for slot in SLOTS:
            decision = pd.Timestamp(f'{date.date()} {slot}')
            cutoff = decision - pd.Timedelta(minutes=15)
            elig = g[g['datetime'] <= cutoff]
            if elig.empty:
                continue
            r = elig.iloc[-1].copy()
            r['snapshot_dt'] = decision
            r['slot'] = slot
            r['symbol'] = symbol
            r['split_like_gap'] = float(split_like)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    keep = ['snapshot_dt','slot','symbol','close','split_like_gap','m15_ret1','m15_ret3','m15_ret10','m15_rsi14','m15_above_ema20','m15_ema20_gt_ema50','m15_macd_delta_pct','m15_atr_pct','m15_vol_ratio20','m15_vwap20_dist','h4_ret','h4_ret3','h4_range_pct','h4_vol_ratio','session_ret','session_range_pos','session_vwap_dist','session_rvol','volume_accel','opening_range_pos']
    return out[keep]


def daily_download(symbols: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(sorted(set(symbols)), start='2018-01-01', auto_adjust=True, actions=False, group_by='ticker', threads=True, progress=False)
    out = {}
    for s in sorted(set(symbols)):
        try:
            x = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            x = x.rename(columns=str.lower).dropna(subset=['close'])
            x.index = pd.to_datetime(x.index).tz_localize(None)
            out[s] = x[['open','high','low','close','volume']].astype(float)
        except Exception:
            out[s] = pd.DataFrame()
    return out


def daily_features(d: pd.DataFrame, prefix='day') -> pd.DataFrame:
    x = d.copy(); c = x['close']; v = x['volume'].fillna(0)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); prev=c.shift(1)
    tr=pd.concat([(x['high']-x['low']),(x['high']-prev).abs(),(x['low']-prev).abs()],axis=1).max(axis=1)
    f=pd.DataFrame(index=x.index)
    f[f'{prefix}_ret1']=c.pct_change(1); f[f'{prefix}_ret5']=c.pct_change(5); f[f'{prefix}_ret20']=c.pct_change(20); f[f'{prefix}_ret63']=c.pct_change(63)
    f[f'{prefix}_rsi14']=rsi(c); f[f'{prefix}_above_ema20']=(c>e20).astype(float); f[f'{prefix}_ema20_gt_ema50']=(e20>e50).astype(float); f[f'{prefix}_ema50_gt_ema200']=(e50>e200).astype(float)
    f[f'{prefix}_macd_delta_pct']=(macd-sig)/c; f[f'{prefix}_atr_pct']=tr.rolling(14).mean()/c; f[f'{prefix}_rv20']=c.pct_change().rolling(20).std(); f[f'{prefix}_vol_ratio20']=v/v.rolling(20).mean().replace(0,np.nan)
    f[f'{prefix}_adv20_log']=np.log1p((c*v).rolling(20).mean())
    return f


def weekly_features(d: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame({
        'open': d['open'].resample('W-FRI').first(), 'high': d['high'].resample('W-FRI').max(),
        'low': d['low'].resample('W-FRI').min(), 'close': d['close'].resample('W-FRI').last(),
        'volume': d['volume'].resample('W-FRI').sum(),
    }).dropna(subset=['close'])
    f = daily_features(w, 'week')
    return f[['week_ret1','week_ret5','week_ret20','week_rsi14','week_above_ema20','week_ema20_gt_ema50','week_macd_delta_pct','week_atr_pct']]


def asof_context(snaps: pd.DataFrame, feature_df: pd.DataFrame, allow_exact=False) -> pd.DataFrame:
    if snaps.empty or feature_df.empty:
        return snaps
    s = snaps.sort_values('snapshot_dt').copy()
    left = pd.DataFrame({'snapshot_dt': s['snapshot_dt'], 'join_date': s['snapshot_dt'].dt.normalize()})
    r = feature_df.reset_index().rename(columns={feature_df.index.name or 'index':'context_date'}).sort_values('context_date')
    m = pd.merge_asof(left.sort_values('join_date'), r, left_on='join_date', right_on='context_date', direction='backward', allow_exact_matches=allow_exact)
    for c in feature_df.columns:
        s[c] = m[c].to_numpy()
    return s


def attach_labels(snaps: pd.DataFrame, stock_bars: pd.DataFrame, spy_bars: pd.DataFrame, day_atr_col='day_atr_pct') -> pd.DataFrame:
    if snaps.empty or stock_bars.empty or spy_bars.empty:
        return snaps
    def closes(b):
        x=b.copy(); x['date']=x['datetime'].dt.normalize(); return x.groupby('date').tail(1).set_index('date')['close'].sort_index()
    sc=closes(stock_bars); pc=closes(spy_bars)
    stock_dates=list(sc.index); spy_dates=set(pc.index)
    pos={d:i for i,d in enumerate(stock_dates)}
    for h in (1,5,10):
        vals=[]
        for _, r in snaps.iterrows():
            d=r['snapshot_dt'].normalize(); i=pos.get(d); cur=float(r['close'])
            if i is None or i+h>=len(stock_dates): vals.append((np.nan,)*6); continue
            fd=stock_dates[i+h]
            if fd not in spy_dates or d not in pc.index: vals.append((np.nan,)*6); continue
            sr=float(sc.loc[fd]/cur-1); pr=float(pc.loc[fd]/pc.loc[d]-1); ex=sr-pr
            atr=float(r.get(day_atr_col)) if pd.notna(r.get(day_atr_col)) else 0.02
            threshold=max(0.003, 0.35*max(0.005,atr)*math.sqrt(h))
            # Extreme moves around split-like events are withheld from labels.
            bad = abs(sr) > 0.55
            vals.append((sr,pr,ex,threshold, np.nan if bad else float(ex>threshold), np.nan if bad else float(ex < -threshold)))
        arr=np.array(vals, dtype=float)
        snaps[f'fwd_{h}_return']=arr[:,0]; snaps[f'spy_fwd_{h}_return']=arr[:,1]; snaps[f'fwd_{h}_excess']=arr[:,2]; snaps[f'move_threshold_{h}']=arr[:,3]; snaps[f'label_up_{h}']=arr[:,4]; snaps[f'label_down_{h}']=arr[:,5]
    return snaps


def score_proxy(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    s += np.where(df['m15_above_ema20']>0.5, 9, -9)
    s += np.where(df['m15_ema20_gt_ema50']>0.5, 8, -8)
    s += np.where(df['m15_macd_delta_pct'].fillna(0)>=0, 6, -6)
    s += np.where(df['session_vwap_dist'].fillna(0)>=0, 6, -6)
    s += np.where(df['h4_ret'].fillna(0)>=0, 9, -9)
    s += np.where(df['day_above_ema20'].fillna(0)>0.5, 7, -7)
    s += np.where(df['day_ema20_gt_ema50'].fillna(0)>0.5, 6, -6)
    s += np.where(df['week_above_ema20'].fillna(0)>0.5, 5, -5)
    s += np.where(df['week_ema20_gt_ema50'].fillna(0)>0.5, 4, -4)
    return s


def main():
    generated = datetime.now(timezone.utc).isoformat()
    failures={}; coverage={}
    intraday={}
    fetch_symbols=UNIVERSE + ['SPY']
    for s in fetch_symbols:
        try:
            intraday[s]=fetch_15m(s)
            if intraday[s].empty: failures[s]='no intraday bars'
        except Exception as e:
            intraday[s]=pd.DataFrame(); failures[s]=f'{type(e).__name__}: {str(e)[:160]}'

    daily = daily_download(UNIVERSE + BENCHMARKS)
    spy_snaps = make_snapshots('SPY', intraday.get('SPY',pd.DataFrame()))
    if spy_snaps.empty:
        raise RuntimeError('SPY intraday history unavailable; cannot build benchmark-relative dataset')
    spy_keep = spy_snaps[['snapshot_dt','m15_ret3','m15_ret10','h4_ret','session_ret','session_vwap_dist','session_rvol']].rename(columns={c:'spy_'+c for c in ['m15_ret3','m15_ret10','h4_ret','session_ret','session_vwap_dist','session_rvol']})

    market_daily={}
    for b in ['SPY','QQQ','IWM','^VIX','TLT','GLD','USO']:
        d=daily.get(b,pd.DataFrame())
        if not d.empty:
            market_daily[b]=daily_features(d, b.replace('^','').lower())

    parts=[]
    for s in UNIVERSE:
        b=intraday.get(s,pd.DataFrame()); d=daily.get(s,pd.DataFrame())
        if b.empty or d.empty:
            continue
        x=make_snapshots(s,b)
        if x.empty: failures[s]=failures.get(s,'no snapshots'); continue
        x=asof_context(x,daily_features(d,'day'),allow_exact=False)
        x=asof_context(x,weekly_features(d),allow_exact=False)
        sec=SECTOR_OF.get(s); sd=daily.get(sec,pd.DataFrame())
        if sec and not sd.empty:
            sf=daily_features(sd,'sector')
            x=asof_context(x,sf[['sector_ret5','sector_ret20','sector_ret63','sector_above_ema20','sector_ema20_gt_ema50']],allow_exact=False)
        for bmk, f in market_daily.items():
            cols=[c for c in f.columns if c.endswith(('ret5','ret20','ret63','atr_pct','rv20')) or 'above_ema20' in c or 'ema20_gt_ema50' in c]
            x=asof_context(x,f[cols],allow_exact=False)
        x=x.merge(spy_keep,on='snapshot_dt',how='left')
        x['rel_m15_10_vs_spy']=x['m15_ret10']-x['spy_m15_ret10']
        x['rel_h4_vs_spy']=x['h4_ret']-x['spy_h4_ret']
        if 'sector_ret20' in x and 'day_ret20' in x: x['rel20_vs_sector']=x['day_ret20']-x['sector_ret20']
        x=attach_labels(x,b,intraday['SPY'])
        x=x[x['split_like_gap']<0.5]
        coverage[s]={'rows':int(len(x)),'first':str(x['snapshot_dt'].min()) if len(x) else None,'last':str(x['snapshot_dt'].max()) if len(x) else None}
        parts.append(x)

    data=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if data.empty:
        raise RuntimeError('no training rows produced')
    data['candidate_score']=score_proxy(data)
    g=data.groupby('snapshot_dt')
    data['candidate_rank_pct']=g['candidate_score'].rank(pct=True)
    data['activity_rank_pct']=g['session_ret'].transform(lambda s:s.abs().rank(pct=True))
    data['rvol_rank_pct']=g['session_rvol'].rank(pct=True)
    data['cross_section_rel_rank']=g['rel_m15_10_vs_spy'].rank(pct=True)
    data=data.sort_values(['snapshot_dt','symbol']).reset_index(drop=True)
    data.to_csv(OUT,index=False,compression='gzip')

    label_stats={}
    for h in (1,5,10):
        label_stats[str(h)]={
            'up_rate':float(data[f'label_up_{h}'].mean(skipna=True)),
            'down_rate':float(data[f'label_down_{h}'].mean(skipna=True)),
            'labeled_rows':int(data[f'label_up_{h}'].notna().sum())
        }
    summary={
        'generated_at':generated,
        'status':'LIVE-LIKE HISTORICAL INTRADAY DATASET BUILT — MODEL TRAINING INPUT, NOT A SIGNAL',
        'dataset':'forecast/data/intraday_live_like_v2.csv.gz',
        'history_source_15m':'Twelve Data 15m; adjust=splits requested; large split-like gaps excluded',
        'daily_context_source':'Yahoo Finance adjusted daily history for lagged Day/Week/regime features',
        'framework':'4 fixed intraday decision times + 15m + rolling 4h + previous completed Day + previous completed Week + SPY intraday + QQQ/IWM/VIX/TLT/GLD/USO + sector context + cross-sectional ranks',
        'decision_times_et':SLOTS,
        'bar_cutoff_rule':'At each decision time, only the most recent fully completed 15-minute bar is used (decision time minus 15 minutes).',
        'lookahead_controls':['Day features use prior completed daily data only','Week features use the last completed weekly bar only','Rolling indicators are past-looking','Labels are stored separately and never used as inputs','Split-like raw-price discontinuities are excluded'],
        'universe_requested':len(UNIVERSE),'symbols_with_rows':len(coverage),'rows':int(len(data)),
        'first_snapshot':str(data['snapshot_dt'].min()),'last_snapshot':str(data['snapshot_dt'].max()),
        'label_definition':'meaningful stock excess return versus SPY; threshold=max(0.3%, 0.35 * prior daily ATR% * sqrt(horizon))',
        'label_stats':label_stats,'coverage':coverage,'failures':failures,
        'validation_policy':'Historical results are development backtests only. Because 2026 has already been inspected during development, final validation for the next model must be prospective/frozen after deployment.'
    }
    SUMMARY.write_text(json.dumps(summary,separators=(',',':')))
    print(json.dumps({k:summary[k] for k in ['status','universe_requested','symbols_with_rows','rows','first_snapshot','last_snapshot','label_stats','failures']},indent=2))

if __name__=='__main__':
    main()
