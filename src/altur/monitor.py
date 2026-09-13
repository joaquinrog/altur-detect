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


def summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Agregados de lo que hay en pantalla. Latencia = trabajo del modelo, no la red."""
    ok = [c for c in calls if c.get("status") == 200]
    ms = sorted(float(c["ms"]) for c in ok if isinstance(c.get("ms"), (int, float)))
    synthetic = sum(1 for c in ok if c.get("is_synthetic") is True)
    human = sum(1 for c in ok if c.get("is_synthetic") is False)
    mb = [float(c["mb"]) for c in calls if isinstance(c.get("mb"), (int, float))]

    def pct(q: float) -> float | None:
        return round(ms[min(int(q * (len(ms) - 1)), len(ms) - 1)], 1) if ms else None

    return {
        "calls": len(calls),
        "ok": len(ok),
        "errors": len(calls) - len(ok),
        "synthetic": synthetic,
        "human": human,
        "inference_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": round(ms[-1], 1) if ms else None},
        "biggest_mb": round(max(mb), 2) if mb else None,
        "last_ts": calls[0].get("ts") if calls else None,
    }
