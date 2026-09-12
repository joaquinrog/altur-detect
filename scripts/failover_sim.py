"""Simulaciones locales del runbook de failover (A4.4). Se EJECUTAN, no se describen.

Cuatro escenarios, cada uno midiendo el tiempo desde el incidente hasta readiness:

    proceso_caido     el proceso muere y se relanza
    bundle_corrupto   un byte cambiado -> readiness 503, nunca una constante silenciosa
    puerto_ocupado    otro proceso tiene el puerto -> el arranque falla de forma legible
    rollback          se vuelve al bundle anterior y /version lo declara

🔴 Esto NO sustituye el ensayo físico. Valida comandos y recuperación del proceso; no valida
DNS, TLS, firewall, carrier, NAT ni la conectividad del juez. Corre sobre el proceso desnudo,
sin Docker, a propósito: aísla la recuperación de la aplicación de la del contenedor, que se
mide aparte en A4.3.

    python scripts/failover_sim.py --bundle models/acoustic_ch0_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEADLINE_S = 40.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str, timeout: float = 2.0) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, None
    except Exception:  # noqa: BLE001 — el servidor todavía no escucha
        return 0, None


def _spawn(port: int, bundle: str | None, **env: str) -> subprocess.Popen:
    e = dict(os.environ, ALTUR_BUNDLE_DIR=bundle or "", PYTHONPATH=str(ROOT / "src"), **env)
    return subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "altur.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
        env=e, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=ROOT,
        start_new_session=True,
    )


def _wait_ready(port: int, *, expect: int = 200, deadline: float = DEADLINE_S) -> float | None:
    """Segundos hasta que readiness responde `expect`. `None` si no llegó."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline:
        code, _ = _get(f"http://127.0.0.1:{port}/health/ready")
        if code == expect:
            return time.perf_counter() - t0
        time.sleep(0.1)
    return None


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def escenario_proceso_caido(bundle: str) -> dict[str, Any]:
    port = _free_port()
    proc = _spawn(port, bundle)
    arranque = _wait_ready(port)
    if arranque is None:
        _kill(proc)
        return {"ok": False, "detalle": "no llegó a ready en el arranque inicial"}

    t0 = time.perf_counter()
    _kill(proc)                                   # el incidente
    caido = _get(f"http://127.0.0.1:{port}/health/ready")[0] == 0

    proc = _spawn(port, bundle)                   # la recuperación (la hace el operador o
    recup = _wait_ready(port)                     # `--restart unless-stopped` en el host)
    total = time.perf_counter() - t0
    _kill(proc)
    return {
        "ok": bool(caido and recup is not None),
        "arranque_a_ready_s": round(arranque, 2),
        "recuperacion_s": round(total, 2),
        "nota": "mide el relanzado del proceso; en el host lo hace `--restart unless-stopped`",
    }


def escenario_bundle_corrupto(bundle: str, tmp: Path) -> dict[str, Any]:
    copia = tmp / "corrupto"
    shutil.rmtree(copia, ignore_errors=True)
    shutil.copytree(bundle, copia)
    victima = copia / "model.json"
    victima.write_bytes(victima.read_bytes() + b" ")   # UN byte

    port = _free_port()
    proc = _spawn(port, str(copia))
    t = _wait_ready(port, expect=503)
    ready_code, _ready_body = _get(f"http://127.0.0.1:{port}/health/ready")
    live_code, _ = _get(f"http://127.0.0.1:{port}/health")
    ver_code, version = _get(f"http://127.0.0.1:{port}/version")
    _kill(proc)

    fuga = version is not None and "/home/" in json.dumps(version)
    return {
        "ok": bool(
            ready_code == 503 and live_code == 200 and ver_code == 200
            and version and version.get("bundle_ok") is False and not fuga
        ),
        "readiness": ready_code,
        "liveness": live_code,
        "tiempo_hasta_503_s": None if t is None else round(t, 2),
        "bundle_error": (version or {}).get("bundle_error"),
        "fuga_de_rutas": fuga,
    }


def escenario_puerto_ocupado(bundle: str) -> dict[str, Any]:
    port = _free_port()
    bloqueador = socket.socket()
    bloqueador.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bloqueador.bind(("127.0.0.1", port))
    bloqueador.listen(1)
    try:
        proc = _spawn(port, bundle)
        try:
            proc.wait(timeout=20)
            salida = (proc.stderr.read() or b"").decode(errors="replace")
        except subprocess.TimeoutExpired:
            _kill(proc)
            return {"ok": False, "detalle": "el servidor NO falló con el puerto ocupado"}
    finally:
        bloqueador.close()
    legible = any(s in salida.lower() for s in ("address already in use", "errno 98"))
    return {
        "ok": bool(proc.returncode != 0 and legible),
        "returncode": proc.returncode,
        "mensaje": next((ln for ln in salida.splitlines() if "address" in ln.lower()), "")[:120],
    }


def escenario_rollback(actual: str, anterior: str) -> dict[str, Any]:
    """Se apunta la configuración al bundle anterior y se comprueba que /version lo dice."""
    port = _free_port()
    proc = _spawn(port, actual)
    _wait_ready(port)
    _, v_actual = _get(f"http://127.0.0.1:{port}/version")
    t0 = time.perf_counter()
    _kill(proc)

    proc = _spawn(port, anterior)
    recup = _wait_ready(port)
    _, v_anterior = _get(f"http://127.0.0.1:{port}/version")
    total = time.perf_counter() - t0
    _kill(proc)
    cambio = (
        v_actual and v_anterior
        and v_actual.get("bundle_created_utc") != v_anterior.get("bundle_created_utc")
    )
    return {
        "ok": bool(recup is not None and cambio),
        "rollback_s": round(total, 2),
        "detector_antes": (v_actual or {}).get("detector"),
        "detector_despues": (v_anterior or {}).get("detector"),
        "created_utc_cambio": bool(cambio),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="models/acoustic_ch0_v1")
    ap.add_argument("--previous", default=None, help="bundle anterior para el rollback")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/altur-failover-sim"))
    args = ap.parse_args(argv)
    args.tmp.mkdir(parents=True, exist_ok=True)

    anterior = args.previous
    if anterior is None:
        # Sin un bundle anterior real se fabrica uno: misma copia, otro `created_utc`, para
        # que el rollback se pueda ENSAYAR. No es un modelo distinto y se declara así.
        anterior = str(args.tmp / "anterior")
        shutil.rmtree(anterior, ignore_errors=True)
        shutil.copytree(args.bundle, anterior)

    resultados = {
        "proceso_caido": escenario_proceso_caido(args.bundle),
        "bundle_corrupto": escenario_bundle_corrupto(args.bundle, args.tmp),
        "puerto_ocupado": escenario_puerto_ocupado(args.bundle),
        "rollback": escenario_rollback(args.bundle, anterior),
    }
    resultados["_alcance"] = (
        "OBS: simulación LOCAL sobre el proceso (sin contenedor). Valida comandos y "
        "recuperación de la aplicación. NO valida DNS, TLS, firewall externo, carrier, NAT, "
        "conectividad del juez ni operación humana real."
    )
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    return 0 if all(v.get("ok") for k, v in resultados.items() if not k.startswith("_")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
