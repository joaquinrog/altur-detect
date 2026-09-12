#!/usr/bin/env python3
"""Prueba de humo contra un `/detect` VIVO — local, VPS o respaldo.

Se corre desde OTRA RED antes de juzgar. Un servidor que pasa los tests unitarios y falla
aquí es el escenario que cuesta la ronda automatizada.

    python scripts/smoke.py --url http://<ip>:8000
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from altur.io import write_wav

OK, BAD = "  ✅", "  ❌"


def _golden(seconds: float = 6.0, sr: int = 8000) -> bytes:
    """Mismo generador que los fixtures: tonos, no audio del dataset."""
    rng = np.random.default_rng(17)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ch0 = rng.normal(0, 0.002, n)
    ch1 = rng.normal(0, 0.002, n)
    for k in range(int(seconds)):
        sl = slice(k * sr, (k + 1) * sr)
        env = np.hanning(sr)
        if k % 2:
            ch0[sl] += 0.30 * env * np.sin(2 * np.pi * 180 * t[sl])
        else:
            ch1[sl] += 0.22 * env * np.sin(2 * np.pi * 320 * t[sl])
    tmp = Path("/tmp/altur_smoke.wav")
    write_wav(str(tmp), ch0.astype(np.float32), ch1.astype(np.float32))
    return tmp.read_bytes()


def _req(url: str, data: bytes | None = None, ctype: str | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if ctype:
        req.add_header("Content-Type", ctype)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body or b"{}")
        except Exception:  # noqa: BLE001 — el smoke reporta lo que sea que respondió el servidor
            parsed = {"raw": body[:200].decode("utf-8", "replace")}
        return e.code, parsed, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n-latency", type=int, default=12)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    wav = _golden()
    b64 = base64.b64encode(wav).decode()
    fails = 0

    print(f"\n== {base} ==\n")

    for path, expect in (("/health", 200), ("/health/ready", 200), ("/version", 200)):
        st, body, ms = _req(f"{base}{path}")
        ok = st == expect
        fails += not ok
        print(f"{OK if ok else BAD} GET {path:16s} {st} {ms:6.1f} ms")
        if path == "/version" and ok:
            for k in ("detector", "git_commit", "bundle_commit", "protocol_id"):
                if k in body:
                    print(f"       {k}: {body[k]}")

    print()
    casos = [
        ("json audio",     json.dumps({"audio": b64}).encode(), "application/json", 200),
        ("json wav alias", json.dumps({"wav": b64}).encode(),   "application/json", 200),
        ("wav crudo",      wav,                                  "audio/wav",        200),
        ("sin campo",      b"{}",                                "application/json", 400),
        ("base64 malo",    json.dumps({"audio": "!!!!"}).encode(), "application/json", 400),
        ("no es wav",      json.dumps({"audio": base64.b64encode(b"x" * 200).decode()}).encode(),
                           "application/json", 400),
    ]
    for nombre, data, ctype, expect in casos:
        st, body, ms = _req(f"{base}/detect", data, ctype)
        ok = st == expect
        fails += not ok
        detalle = (
            f"is_synthetic={body.get('is_synthetic')} confidence={body.get('confidence')}"
            if st == 200 else f"error={body.get('error')}"
        )
        print(f"{OK if ok else BAD} POST /detect {nombre:14s} {st} {ms:6.1f} ms  {detalle}")
        if st == 200 and set(body) - {"is_synthetic", "confidence", "seconds_used", "degraded"}:
            print(f"       ⚠️  campos inesperados: {set(body)}")

    print()
    tiempos = [_req(f"{base}/detect", json.dumps({"audio": b64}).encode(),
                    "application/json")[2] for _ in range(args.n_latency)]
    tiempos.sort()
    p50 = statistics.median(tiempos)
    p95 = tiempos[min(len(tiempos) - 1, int(0.95 * len(tiempos)))]
    print(f"  latencia n={len(tiempos)}  p50 {p50:.1f} ms  p95 {p95:.1f} ms  max {max(tiempos):.1f} ms")
    print("  ⚠️  audio de prueba de 6 s; el presupuesto real es <1 s para 150 s de audio\n")

    print("RESULTADO:", "todo OK" if fails == 0 else f"{fails} fallo(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
