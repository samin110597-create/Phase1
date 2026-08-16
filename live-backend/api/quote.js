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

async function finnhubQuote(symbol, token) {
  const u = new URL('https://finnhub.io/api/v1/quote');
  u.searchParams.set('symbol', symbol);
  u.searchParams.set('token', token);
  const r = await fetch(u, { cache: 'no-store' });
  if (!r.ok) throw new Error(`Finnhub HTTP ${r.status}`);
  const d = await r.json();
  if (!(Number(d.c) > 0)) throw new Error('No usable Finnhub quote');
  const ts = Number(d.t) || null;
  return {
    symbol,
    price: Number(d.c),
    previous_close: Number(d.pc) || null,
    day_high: Number(d.h) || null,
    day_low: Number(d.l) || null,
    day_open: Number(d.o) || null,
    timestamp: ts,
    age_seconds: ts ? Math.max(0, Math.round(Date.now() / 1000 - ts)) : null,
    provider: 'Finnhub'
  };
}

export default async function handler(req, res) {
  setCors(res);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'GET only' });

  const token = process.env.FINNHUB_API_KEY;
  if (!token) return res.status(503).json({ error: 'FINNHUB_API_KEY is not configured on the live backend' });

  const symbols = cleanSymbols(req.query.symbols || req.query.symbol);
  if (!symbols.length) return res.status(400).json({ error: 'Provide 1-3 symbols' });

  const rows = await Promise.all(symbols.map(async symbol => {
    try { return await finnhubQuote(symbol, token); }
    catch (e) { return { symbol, error: e.message }; }
  }));

  return res.status(200).json({
    generated_at: new Date().toISOString(),
    mode: 'secure live pull; browser receives no provider API keys',
    symbols_requested: symbols,
    rows
  });
}
