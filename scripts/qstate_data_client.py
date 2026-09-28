from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://stock-truth-v2.samin110597.deno.net"


def _base() -> str:
    return os.getenv("QSTATE_API_BASE", DEFAULT_BASE).rstrip("/")


def _get(path: str, params: dict[str, str] | None = None, timeout: int = 30) -> dict:
    url = _base() + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Phase1-QState-Research/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{data.get('error')}: {data.get('message') or ''}".strip())
    return data


def health() -> dict:
    return _get("/health")


def quote(symbol: str) -> dict:
    return _get("/v1/quote", {"symbol": symbol.upper().strip()})


def market(symbol: str, timeframe: str = "1D") -> dict:
    timeframe = timeframe.upper().strip()
    if timeframe not in {"15M", "1H", "4H", "1D"}:
        raise ValueError("timeframe must be one of 15M, 1H, 4H, 1D")
    return _get("/v1/market", {"symbol": symbol.upper().strip(), "timeframe": timeframe}, timeout=60)


if __name__ == "__main__":
    print(json.dumps(health(), indent=2))
