const ALLOWED_ORIGIN = 'https://samin110597-create.github.io';

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', ALLOWED_ORIGIN);
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'no-store, max-age=0');
}

function cleanSymbols(raw) {
  return [...new Set(String(raw || '')
    .toUpperCase()
    .split(',')
    .map(s => s.trim())
    .filter(s => /^[A-Z][A-Z0-9.-]{0,9}$/.test(s)))]
    .slice(0, 3);
}

function clamp(x, lo = 0, hi = 100) { return Math.max(lo, Math.min(hi, x)); }
function pct(a, b) { return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? (a / b - 1) * 100 : null; }
function last(a) { return a?.length ? a[a.length - 1] : null; }
function mean(a) { const z = a.filter(Number.isFinite); return z.length ? z.reduce((s,x)=>s+x,0) / z.length : null; }

function ema(values, span) {
  if (!values.length) return [];
  const k = 2 / (span + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) out.push(values[i] * k + out[i - 1] * (1 - k));
  return out;
}

function rsi(values, n = 14) {
  if (values.length <= n) return null;
  let gain = 0, loss = 0;
  for (let i = 1; i <= n; i++) {
    const d = values[i] - values[i - 1];
    if (d >= 0) gain += d; else loss -= d;
  }
  gain /= n; loss /= n;
  for (let i = n + 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    gain = (gain * (n - 1) + Math.max(d, 0)) / n;
    loss = (loss * (n - 1) + Math.max(-d, 0)) / n;
  }
  if (loss === 0) return 100;
  const rs = gain / loss;
  return 100 - 100 / (1 + rs);
}

function atr(bars, n = 14) {
  if (bars.length < n + 1) return null;
  const tr = [];
  for (let i = 1; i < bars.length; i++) {
    const b = bars[i], p = bars[i - 1];
    tr.push(Math.max(b.high - b.low, Math.abs(b.high - p.close), Math.abs(b.low - p.close)));
  }
  return mean(tr.slice(-n));
}

function normalizeBars(values) {
  return (Array.isArray(values) ? values : []).map(x => ({
    datetime: x.datetime,
    open: Number(x.open), high: Number(x.high), low: Number(x.low), close: Number(x.close), volume: Number(x.volume || 0)
  })).filter(x => Number.isFinite(x.close)).sort((a,b) => String(a.datetime).localeCompare(String(b.datetime)));
}

function tfFeatures(bars, label) {
  const closes = bars.map(x => x.close), vols = bars.map(x => x.volume || 0);
  if (closes.length < 55) return null;
  const e12 = ema(closes, 12), e26 = ema(closes, 26), e20 = ema(closes, 20), e50 = ema(closes, 50);
  const macd = e12.map((x,i) => x - e26[i]);
  const sig = ema(macd, 9);
  const c = last(closes), v20 = mean(vols.slice(-20));
  const typical = bars.map(x => (x.high + x.low + x.close) / 3);
  let pv = 0, vv = 0;
  for (let i = Math.max(0, bars.length - 20); i < bars.length; i++) { pv += typical[i] * vols[i]; vv += vols[i]; }
  const vw = vv ? pv / vv : null;
  const a = atr(bars, 14);
  return {
    label,
    last_bar: last(bars)?.datetime || null,
    close: c,
    return_3bars_pct: pct(c, closes[closes.length - 4]),
    return_10bars_pct: pct(c, closes[closes.length - 11]),
    rsi14: rsi(closes, 14),
    ema20: last(e20),
    ema50: last(e50),
    above_ema20: c > last(e20),
    ema20_above_ema50: last(e20) > last(e50),
    macd_delta_pct: c ? ((last(macd) - last(sig)) / c) * 100 : null,
    atr_pct: c && a ? a / c * 100 : null,
    volume_ratio20: v20 ? last(vols) / v20 : null,
    rolling_vwap20_distance_pct: c && vw ? (c / vw - 1) * 100 : null
  };
}

function rolling4hFrom15m(bars) {
  const closes = bars.map(x => x.close), vols = bars.map(x => x.volume || 0);
  if (bars.length < 80) return null;
  const c = last(closes);
  const v16 = mean(vols.slice(-16)), v80 = mean(vols.slice(-80));
  const hi = Math.max(...bars.slice(-16).map(x => x.high));
  const lo = Math.min(...bars.slice(-16).map(x => x.low));
  return {
    label: '4h rolling',
    last_bar: last(bars)?.datetime || null,
    return_4h_pct: pct(c, closes[closes.length - 17]),
    return_12h_pct: pct(c, closes[closes.length - 49]),
    volume_ratio: v80 ? v16 / v80 : null,
    range_pct: c ? (hi - lo) / c * 100 : null
  };
}

function weekKey(dateString) {
  const d = new Date(`${dateString.slice(0,10)}T00:00:00Z`);
  const day = d.getUTCDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setUTCDate(d.getUTCDate() + diff);
  return d.toISOString().slice(0,10);
}

function aggregateWeekly(daily) {
  const groups = new Map();
  for (const b of daily) {
    const k = weekKey(b.datetime);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(b);
  }
  return [...groups.entries()].sort((a,b)=>a[0].localeCompare(b[0])).map(([k,a]) => ({
    datetime: k,
    open: a[0].open,
    high: Math.max(...a.map(x=>x.high)),
    low: Math.min(...a.map(x=>x.low)),
    close: last(a).close,
    volume: a.reduce((s,x)=>s+(x.volume||0),0)
  }));
}

async function twelveSeries(symbol, interval, outputsize, key) {
  const u = new URL('https://api.twelvedata.com/time_series');
  u.searchParams.set('symbol', symbol);
  u.searchParams.set('interval', interval);
  u.searchParams.set('outputsize', String(outputsize));
  u.searchParams.set('timezone', 'America/New_York');
  u.searchParams.set('apikey', key);
  const r = await fetch(u, { cache: 'no-store' });
  if (!r.ok) throw new Error(`Twelve Data HTTP ${r.status}`);
  const d = await r.json();
  if (d.status === 'error' || !Array.isArray(d.values)) throw new Error(d.message || 'No Twelve Data bars');
  return normalizeBars(d.values);
}

async function finnhubQuote(symbol, token) {
  const u = new URL('https://finnhub.io/api/v1/quote');
  u.searchParams.set('symbol', symbol); u.searchParams.set('token', token);
  const r = await fetch(u, { cache: 'no-store' });
  if (!r.ok) throw new Error(`Finnhub HTTP ${r.status}`);
  const d = await r.json();
  const price = Number(d.c), ts = Number(d.t) || null;
  if (!(price > 0)) throw new Error('No usable live quote');
  return { price, timestamp: ts, age_seconds: ts ? Math.max(0, Math.round(Date.now()/1000-ts)) : null, provider: 'Finnhub' };
}

function marketStateNow() {
  const p = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday:'short', hour:'2-digit', minute:'2-digit', hour12:false }).formatToParts(new Date());
  const o = Object.fromEntries(p.map(x=>[x.type,x.value]));
  const mins = Number(o.hour) * 60 + Number(o.minute);
  const open = !['Sat','Sun'].includes(o.weekday) && mins >= 570 && mins < 960;
  return { market_open: open, ny_weekday: o.weekday, ny_time: `${o.hour}:${o.minute}` };
}

function signedLiveScore(x, spy) {
  let s = 0; const why = [];
  const m = x.m15, h = x.h4, d = x.day, w = x.week;
  function add(cond, pts, pos, neg) { if (cond === true) { s += pts; why.push(pos); } else if (cond === false) { s -= pts; why.push(neg); } }
  if (m) {
    add(m.above_ema20, 8, '15m above EMA20', '15m below EMA20');
    add(m.ema20_above_ema50, 7, '15m EMA stack positive', '15m EMA stack negative');
    if (Number.isFinite(m.rsi14)) { if (m.rsi14 >= 55 && m.rsi14 <= 75) { s += 5; why.push('15m RSI supportive'); } else if (m.rsi14 <= 45) { s -= 5; why.push('15m RSI weak'); } }
    if (Number.isFinite(m.macd_delta_pct)) { s += m.macd_delta_pct >= 0 ? 5 : -5; why.push(m.macd_delta_pct >= 0 ? '15m MACD positive' : '15m MACD negative'); }
    if (Number.isFinite(m.rolling_vwap20_distance_pct)) { s += m.rolling_vwap20_distance_pct >= 0 ? 5 : -5; why.push(m.rolling_vwap20_distance_pct >= 0 ? 'above rolling VWAP' : 'below rolling VWAP'); }
  }
  if (h && Number.isFinite(h.return_4h_pct)) { s += h.return_4h_pct >= 0 ? 8 : -8; why.push(h.return_4h_pct >= 0 ? '4h momentum positive' : '4h momentum negative'); }
  if (d) { add(d.above_ema20, 7, 'daily above EMA20', 'daily below EMA20'); add(d.ema20_above_ema50, 6, 'daily trend positive', 'daily trend negative'); }
  if (w) { add(w.above_ema20, 5, 'weekly above EMA20', 'weekly below EMA20'); add(w.ema20_above_ema50, 4, 'weekly trend positive', 'weekly trend negative'); }
  if (spy?.m15 && m && Number.isFinite(m.return_10bars_pct) && Number.isFinite(spy.m15.return_10bars_pct)) {
    const rel = m.return_10bars_pct - spy.m15.return_10bars_pct;
    s += rel >= 0 ? 7 : -7; why.push(rel >= 0 ? 'beating SPY intraday' : 'lagging SPY intraday');
  }
  const up = clamp(50 + s), down = clamp(50 - s);
  return { signed_edge: s, upside_score: up, downside_score: down, bias: s >= 12 ? 'BULLISH' : s <= -12 ? 'BEARISH' : 'NEUTRAL', reasons: why.slice(0,8) };
}

async function buildSymbol(symbol, twelveKey, finnhubKey) {
  const [m15, day, quote] = await Promise.all([
    twelveSeries(symbol, '15min', 220, twelveKey),
    twelveSeries(symbol, '1day', 260, twelveKey),
    finnhubQuote(symbol, finnhubKey)
  ]);
  const weekBars = aggregateWeekly(day);
  return { symbol, quote, m15: tfFeatures(m15, '15m'), h4: rolling4hFrom15m(m15), day: tfFeatures(day, 'day'), week: tfFeatures(weekBars, 'week') };
}

export default async function handler(req, res) {
  setCors(res);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'GET only' });
  const twelveKey = process.env.TWELVE_DATA_API_KEY, finnhubKey = process.env.FINNHUB_API_KEY;
  if (!twelveKey || !finnhubKey) return res.status(503).json({ error: 'Live backend requires TWELVE_DATA_API_KEY and FINNHUB_API_KEY' });
  const symbols = cleanSymbols(req.query.symbols || req.query.symbol);
  if (!symbols.length) return res.status(400).json({ error: 'Provide 1-3 symbols' });

  const requested = [...new Set([...symbols, 'SPY'])];
  const built = {};
  await Promise.all(requested.map(async s => {
    try { built[s] = await buildSymbol(s, twelveKey, finnhubKey); }
    catch (e) { built[s] = { symbol: s, error: e.message }; }
  }));
  const spy = built.SPY?.error ? null : built.SPY;
  const state = marketStateNow();
  const rows = symbols.map(s => {
    const x = built[s];
    if (!x || x.error) return x || { symbol:s, error:'No data' };
    const score = signedLiveScore(x, spy);
    const liveStale = state.market_open && Number.isFinite(x.quote?.age_seconds) && x.quote.age_seconds > 120;
    return { ...x, ...score, live_stale: !!liveStale, forecast_status: !state.market_open ? 'MARKET_CLOSED' : liveStale ? 'STALE_LIVE_QUOTE' : 'PROVISIONAL_LIVE' };
  });

  return res.status(200).json({
    generated_at: new Date().toISOString(),
    engine: 'Phase1 Live Forecast Engine V2',
    status: 'PROVISIONAL — LIVE-DATA ENGINE, NOT CALIBRATED PROBABILITY',
    framework: 'Finnhub live quote + Twelve Data 15m + rolling 4h + daily + weekly + SPY context',
    market: state,
    refresh_guidance: { quote_seconds: 10, forecast_seconds: 60, max_symbols: 3 },
    rows
  });
}
