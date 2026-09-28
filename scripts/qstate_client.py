"""Shared Q-State data client for Phase1 challenger experiments.

Phase1 must not build a second production provider stack. This module reads the
same canonical Deno market/research service used by Q-State Unified.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

DEFAULT_BASE = "https://stock-truth-v2.samin110597.deno.net"


def _get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Phase1-QState-Challenger/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_base() -> str:
    return os.environ.get("QSTATE_API_BASE", DEFAULT_BASE).rstrip("/")


def health() -> dict:
    return _get(api_base() + "/health", 20)


def market(symbol: str, timeframe: str = "1D") -> dict:
    q = urllib.parse.urlencode({"symbol": symbol.upper().strip(), "timeframe": timeframe.upper().strip()})
    return _get(api_base() + "/v1/market?" + q, 45)


def research(symbol: str) -> dict:
    q = urllib.parse.urlencode({"symbol": symbol.upper().strip()})
    try:
        return _get(api_base() + "/v1/research?" + q, 30)
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)[:240]}


def canonical_snapshot(symbol: str, timeframe: str = "1D") -> dict:
    return {
        "source": "Q-State Unified Deno",
        "health": health(),
        "market": market(symbol, timeframe),
        "research": research(symbol),
        "production_weight": 0,
        "policy": "Phase1 is a challenger lab. This snapshot may be used for experiments, but only Q-State Unified can produce the production forecast.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--timeframe", default="1D")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        h = health()
        m = market(args.symbol, args.timeframe)
        bars = ((m.get("primary") or {}).get("bars") or [])
        if h.get("status") != "OK" or len(bars) < 80:
            raise SystemExit("Q-State canonical source smoke check failed")
        print(json.dumps({
            "status": "PASS",
            "service": h.get("service"),
            "canonical_model": h.get("canonical_model") or "Q-State Unified",
            "symbol": args.symbol.upper(),
            "timeframe": args.timeframe.upper(),
            "bars": len(bars),
            "provider": (m.get("primary") or {}).get("provider"),
        }, indent=2))
        return

    print(json.dumps(canonical_snapshot(args.symbol, args.timeframe), indent=2))


if __name__ == "__main__":
    main()
