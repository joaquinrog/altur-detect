"""Registro de llamadas para el Monitor de la mesa. Apagado por defecto.

Tres reglas, en este orden:

1. **`/detect` manda.** Nada de aquí puede tumbar, retrasar ni cambiar la respuesta del
   endpoint que califican. Todo lo que se llama desde el camino de inferencia va envuelto
   y devuelve `None` en vez de propagar.
2. **Apagado por defecto.** Sin `ALTUR_MONITOR=1` no se registra nada y las rutas del
   monitor responden 404, como si no existieran.
3. **Nada sensible.** Del `call_id` se guardan 8 caracteres de su SHA-256, que sirven para
   distinguir llamadas y no para reconstruirlas. Ni audio, ni cuerpo, ni cabeceras.

**Por qué un archivo y no una lista en memoria:** el servicio corre con varios workers de
uvicorn (4 en Vultr). Un anillo en memoria vive en un proceso, así que el worker que
atiende `/monitor/calls` solo vería su propia parte de las llamadas — un cuarto, repartido
al azar. Un JSONL en el tmpfs lo escriben todos y lo lee cualquiera.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = "/tmp/altur-monitor.jsonl"  # tmpfs del contenedor: efímero a propósito
MAX_BYTES = 1 << 20  # 1 MB: unas 5 000 llamadas. Al pasarse se conserva la cola.
DEFAULT_LIMIT = 100


def path() -> Path:
    return Path(os.environ.get("ALTUR_MONITOR_PATH") or DEFAULT_PATH)


def call_ref(call_id: Any) -> str | None:
    """8 caracteres del SHA-256 del `call_id`: distingue llamadas sin conservar el id."""
    if not isinstance(call_id, str) or not call_id:
        return None
    return hashlib.sha256(call_id.encode("utf-8", "replace")).hexdigest()[:8]


def record(**fields: Any) -> None:
    """Añade una línea. Nunca lanza: si el monitor falla, `/detect` ni se entera."""
    try:
        p = path()
        line = json.dumps({"ts": time.time(), **fields}, ensure_ascii=False, default=str)
        with p.open("a", encoding="utf-8") as f:  # O_APPEND: una línea corta no se entrelaza
            f.write(line + "\n")
        if p.stat().st_size > MAX_BYTES:
            _truncate(p)
    except Exception:  # noqa: BLE001 — el monitor es accesorio; el endpoint no
        return


def _truncate(p: Path) -> None:
    try:
        tail = p.read_text(encoding="utf-8", errors="replace").splitlines()[-DEFAULT_LIMIT:]
        tmp = p.with_suffix(".tmp")
        tmp.write_text("\n".join(tail) + "\n", encoding="utf-8")
        tmp.replace(p)  # atómico: nadie lee un archivo a medio escribir
    except Exception:  # noqa: BLE001
        return


def read(limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Las últimas `limit` llamadas, de la más nueva a la más vieja. Nunca lanza."""
    try:
        lines = path().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw in reversed(lines):
        if len(out) >= limit:
            break
        try:
            item = json.loads(raw)
        except ValueError:
            continue  # línea a medias por una escritura simultánea: se ignora, no se rompe
        if isinstance(item, dict):
            out.append(item)
    return out


BUDGET_S = 30.0  # el presupuesto del juez por llamada (README de Altur, contrato `429adf7`)


def _num(c: dict[str, Any], k: str) -> float | None:
    v = c.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def total_ms(c: dict[str, Any]) -> float | None:
    """Lo que gasta la llamada del presupuesto de 30 s: todo el trabajo del servidor.

    Es `server_ms`, no `upload_ms + ms`. Entre esos dos queda el parseo del JSON, el
    base64 y el WAV, que no están en ninguno y medidos son decenas de ms sobre un cuerpo
    de 6 MB. Sumar solo los dos daba un total corto a la mitad.

    Sigue sin ser exactamente lo que cronometra el cliente del juez: le faltan el
    handshake y el viaje de vuelta (~2 RTT), así que se queda un poco por debajo.
    """
    srv = _num(c, "server_ms")
    if srv is not None:
        return srv
    up, ms = _num(c, "upload_ms"), _num(c, "ms")
    if up is None and ms is None:
        return None
    return (up or 0.0) + (ms or 0.0)


def decode_ms(c: dict[str, Any]) -> float | None:
    """El hueco entre la subida y la inferencia: parseo JSON + base64 + WAV."""
    srv, up, ms = _num(c, "server_ms"), _num(c, "upload_ms"), _num(c, "ms")
    if srv is None or up is None or ms is None:
        return None
    return max(0.0, srv - up - ms)


def _pcts(values: list[float]) -> dict[str, float | None]:
    v = sorted(values)
    if not v:
        return {"p50": None, "p95": None, "max": None}

    def at(q: float) -> float:
        return round(v[min(int(q * (len(v) - 1)), len(v) - 1)], 1)

    return {"p50": at(0.5), "p95": at(0.95), "max": round(v[-1], 1)}


def summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Agregados de lo que hay en pantalla.

    Cuatro cosas distintas que no hay que confundir. La llamada se parte en tres etapas
    que suman el total: `upload_ms` (el cliente subiendo el cuerpo), `decode_ms` (parsear
    el JSON, el base64 y el WAV) e `inference_ms` (el modelo, lo mismo que
    `X-Inference-Ms`). `total_ms` es el trabajo entero del servidor, y es el que se compara
    contra los 30 s del juez.
    """
    ok = [c for c in calls if c.get("status") == 200]
    ms = [v for c in ok if (v := _num(c, "ms")) is not None]
    ups = [v for c in calls if (v := _num(c, "upload_ms")) is not None]
    decs = [v for c in calls if (v := decode_ms(c)) is not None]
    totals = [v for c in calls if (v := total_ms(c)) is not None]
    mb = [v for c in calls if (v := _num(c, "mb")) is not None]

    # El desglose de LA PEOR llamada, no el máximo de cada etapa por separado: esos máximos
    # salen de llamadas distintas y no suman el total, que es justo lo que promete la
    # cabecera de la página.
    worst_call = max(calls, key=lambda c: total_ms(c) or -1.0) if totals else None
    peor = total_ms(worst_call) if worst_call else None
    return {
        "calls": len(calls),
        "ok": len(ok),
        "errors": len(calls) - len(ok),
        "synthetic": sum(1 for c in ok if c.get("is_synthetic") is True),
        "human": sum(1 for c in ok if c.get("is_synthetic") is False),
        "inference_ms": _pcts(ms),
        "upload_ms": _pcts(ups),
        "decode_ms": _pcts(decs),
        "total_ms": _pcts(totals),
        "budget_s": BUDGET_S,
        # Cuánto margen sobra en la PEOR llamada: el número que importa, porque el juez
        # cuenta como fallo cada llamada que se pase, no el promedio.
        "worst_headroom_x": round(BUDGET_S * 1000.0 / peor, 1) if peor else None,
        "worst": {
            "total_ms": round(peor, 1),
            "upload_ms": round(_num(worst_call, "upload_ms") or 0.0, 1),
            "decode_ms": round(decode_ms(worst_call) or 0.0, 1),
            "inference_ms": round(_num(worst_call, "ms") or 0.0, 1),
            "mb": _num(worst_call, "mb"),
        } if worst_call and peor else None,
        "biggest_mb": round(max(mb), 2) if mb else None,
        "last_ts": calls[0].get("ts") if calls else None,
    }
