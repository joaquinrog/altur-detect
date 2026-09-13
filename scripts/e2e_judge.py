"""Prueba `/detect` como lo hará el juez de Altur, contra un contenedor limpio o una URL.

Corre el cliente oficial `scripts/check_endpoint.py` de `alturio/hackmty26@429adf7` **sin
modificar** (se verifica su SHA-256) y añade lo que ese cliente no cubre:

- la llamada más larga del set elegido (el cuerpo más grande; en `train` son 11.7 MB en base64,
  más del doble de los "~5 MB" del README oficial);
- entradas inválidas, que deben ser 400 y nunca 500;
- en modo contenedor, límites de 2 CPU y 8 GB como la instancia de Vultr (`voc-c-4c-8gb`).

Mide el servidor desde la misma máquina: no reproduce la subida por la red del venue (D-A6.2).

    # contenedor limpio con un bundle montado
    python scripts/e2e_judge.py --image altur-detect:e2e --bundle models/spectral_factory_lfcc_v1
    # cualquier servidor ya levantado (p. ej. C1, que por GPL no entra a la imagen)
    python scripts/e2e_judge.py --url http://127.0.0.1:18011/detect --label c1
    # un set propio con el mismo formato de manifest (anon_id,label,split,duration_s)
    python scripts/e2e_judge.py --url ... --manifest set/manifest.csv --audio-dir set/audio --split hidden --n 0

`val` exige `--use-val` y deja una fila en `experiments/val_looks.csv` (AGENTS.md, regla 4).
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "data" / "official" / "check_endpoint.py"
CHECKER_SHA256 = "593f78ceb80017e791f0f8d552ca6a7b3b6763c11beedfa1f68ab1a55c363364"
VAL_LOOKS = ROOT / "experiments" / "val_looks.csv"
JUDGE_TIMEOUT_S = 30.0
CPUS, MEMORY = "2", "8g"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(url: str, body: bytes | None = None, content_type: str = "application/json",
            timeout: float = JUDGE_TIMEOUT_S) -> tuple[int, bytes, float]:
    req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    if body is not None:
        req.add_header("Content-Type", content_type)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        # Por red lenta el cuerpo puede no terminar de subir en 30 s: para el juez eso es una
        # llamada fallida, no un motivo para abortar el resto de la prueba. Status 0 = sin respuesta.
        return 0, repr(e).encode(), time.perf_counter() - t0


def wait_ready(base: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if request(f"{base}/health/ready", timeout=2)[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def official_body(call_id: str, wav: bytes) -> bytes:
    return json.dumps({
        "call_id": call_id, "audio_base64": base64.b64encode(wav).decode(),
        "sample_rate": 8000, "channels": 2,
    }).encode()


def valid_answer(status: int, raw: bytes) -> bool:
    if status != 200:
        return False
    try:
        body = json.loads(raw)
    except ValueError:
        return False
    conf = body.get("confidence")
    return isinstance(body.get("is_synthetic"), bool) and (
        conf is None or (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0)
    )


def probes(url: str, longest: dict, audio_dir: Path) -> list[dict]:
    """Casos que el cliente oficial no manda. Cada uno declara lo que se espera."""
    out = []
    wav = (audio_dir / f"{longest['anon_id']}.wav").read_bytes()
    body = official_body("call_E2ELONGEST", wav)
    status, raw, secs = request(url, body)
    out.append({
        "case": "llamada_mas_larga", "duration_s": float(longest["duration_s"]),
        "body_mb": round(len(body) / 1e6, 2), "status": status, "latency_s": round(secs, 3),
        "ok": valid_answer(status, raw) and secs < JUDGE_TIMEOUT_S,
    })
    invalid = [
        ("json_corrupto", b"{no es json"),
        ("sin_audio", json.dumps({"call_id": "call_E2E0001", "sample_rate": 8000, "channels": 2}).encode()),
        ("base64_invalido", json.dumps({"call_id": "call_E2E0002", "audio_base64": "!!!!",
                                        "sample_rate": 8000, "channels": 2}).encode()),
        ("wav_invalido", official_body("call_E2E0003", b"RIFF0000WAVEnope")),
    ]
    for case, payload in invalid:
        status, _raw, secs = request(url, payload)
        out.append({"case": case, "status": status, "latency_s": round(secs, 3), "ok": status == 400})
    return out


def record_val_look(label: str, n: int) -> None:
    with VAL_LOOKS.open("a", newline="") as f:
        csv.writer(f).writerow([
            dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"e2e_judge_{label}_val_n{n}", "e2e_judge.py",
            "e2e con el cliente oficial sobre val (--use-val); no se ajusta nada con el resultado",
        ])


def start_container(args, port: int) -> str:
    name = f"altur-e2e-{os.getpid()}"
    cmd = [args.docker, "run", "-d", "--rm", "--name", name, "--cpus", CPUS, "--memory", MEMORY,
           "-p", f"127.0.0.1:{port}:8000"]
    if args.bundle:
        cmd += ["-v", f"{args.bundle.resolve()}:/app/models/e2e:ro", "-e", "ALTUR_BUNDLE_DIR=/app/models/e2e"]
    subprocess.run(cmd + [args.image], check=True, capture_output=True)
    return name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="un /detect ya levantado")
    target.add_argument("--image", help="imagen a correr limpia con límites de Vultr")
    ap.add_argument("--bundle", type=Path, help="bundle a montar en el contenedor (default: el de la imagen)")
    ap.add_argument("--docker", default="docker")
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "audio")
    ap.add_argument("--split", default="train", choices=["train", "val", "all", "hidden"])
    ap.add_argument("--n", type=int, default=40, help="llamadas del cliente oficial (0 = todas)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-val", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "runs" / "e2e")
    args = ap.parse_args(argv)

    if args.split in ("val", "all") and not args.use_val:
        print("este split incluye val: pásalo con --use-val (deja fila en val_looks.csv)", file=sys.stderr)
        return 2
    if not CHECKER.exists() or sha256(CHECKER) != CHECKER_SHA256:
        print(f"falta el cliente oficial verificado en {CHECKER}: corre `make data`",
              file=sys.stderr)
        return 2

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    pool = rows if args.split == "all" else [r for r in rows if r["split"] == args.split]
    if not pool:
        print(f"no hay filas con split={args.split!r} en {args.manifest}", file=sys.stderr)
        return 2
    longest = max(pool, key=lambda r: float(r["duration_s"]))

    label = args.label or (args.bundle.name if args.bundle else Path(args.image or "url").name)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calls_json = args.out_dir / f"{stamp}_{label}_calls.json"

    container = None
    if args.image:
        container = start_container(args, args.port)
        base = f"http://127.0.0.1:{args.port}"
        url = f"{base}/detect"
    else:
        url = args.url
        base = url.rsplit("/detect", 1)[0]
    try:
        if args.image and not wait_ready(base, 120):
            logs = subprocess.run([args.docker, "logs", container], capture_output=True, text=True, check=False)
            print("el contenedor no llegó a ready:\n" + logs.stdout[-2000:] + logs.stderr[-2000:], file=sys.stderr)
            return 1
        version = {}
        if args.image:
            status, raw, _ = request(f"{base}/version", timeout=5)
            version = json.loads(raw) if status == 200 else {}

        if args.split in ("val", "all"):
            record_val_look(label, args.n)

        checks = probes(url, longest, args.audio_dir)
        proc = subprocess.run(
            [sys.executable, str(CHECKER), "--url", url, "--manifest", str(args.manifest),
             "--audio-dir", str(args.audio_dir), "--split", args.split, "--n", str(args.n),
             "--seed", str(args.seed), "--timeout", str(JUDGE_TIMEOUT_S), "--out", str(calls_json)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0 or not calls_json.exists():
            print(proc.stdout[-3000:] + proc.stderr[-3000:], file=sys.stderr)
            return 1
        official = json.loads(calls_json.read_text(encoding="utf-8"))
    finally:
        if container:
            subprocess.run([args.docker, "rm", "-f", container], capture_output=True, check=False)

    summary = official["summary"]
    latencies = sorted(r["latency_s"] for r in official["results"] if "latency_s" in r)
    report = {
        "label": label,
        "target": {"url": url, "image": args.image, "bundle": str(args.bundle) if args.bundle else None,
                   "limits": {"cpus": CPUS, "memory": MEMORY} if args.image else None,
                   "detector": version.get("detector")},
        "checker_sha256": CHECKER_SHA256,
        "split": args.split, "n": args.n, "seed": args.seed,
        "official_summary": summary,
        "latency_s": {"p50": statistics.median(latencies),
                      "p95": latencies[int(0.95 * (len(latencies) - 1))], "max": latencies[-1]},
        "probes": checks,
    }
    ok = (summary["errors"] == 0 and summary["answered"] == summary["calls"]
          and report["latency_s"]["max"] < JUDGE_TIMEOUT_S and all(p["ok"] for p in checks))
    report["passed"] = ok
    (args.out_dir / f"{stamp}_{label}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("label", "passed", "latency_s")} | {
        "official": {k: summary.get(k) for k in ("calls", "answered", "errors", "balanced_accuracy", "auc")},
        "probes": [(p["case"], p["status"], p["ok"]) for p in checks],
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
