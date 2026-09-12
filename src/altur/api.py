"""`POST /detect` — el endpoint que califican los jueces.

Importa SOLO `detector`/`bundle`. Cambiar de enfoque = cambiar el bundle en disco.
Este archivo no debería volver a tocarse cuando cambie el modelo.

Política de entrada (plan §7): primero el contrato oficial EXACTO; la tolerancia después
y nunca inventando semántica. Por defecto la respuesta lleva solo los dos campos del
contrato — los extras se activan con ALTUR_RESPONSE_EXTRAS=1 y solo si Altur confirma
que los tolera.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from . import bundle
from .io import AudioDecodeError, decode, decode_base64
from .types import Prediction

log = logging.getLogger("altur.api")

# Nombres plausibles del campo del payload. El PDF no lo fija (UNK, notes/01 §6);
# aceptar alias es seguro barato contra el modo de falla que mata la ronda automatizada.
AUDIO_FIELDS = ("audio", "audio_base64", "audio_data", "wav", "wav_base64", "data", "b64", "file")


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        self.bundle_dir = os.environ.get("ALTUR_BUNDLE_DIR") or None
        # Mono NUNCA se duplica a estéreo. allow_mono=True => detector degradado ch0-only.
        self.allow_mono = _env_flag("ALTUR_ALLOW_MONO", True)
        self.allow_resample = _env_flag("ALTUR_ALLOW_RESAMPLE", True)
        self.response_extras = _env_flag("ALTUR_RESPONSE_EXTRAS", False)
        self.max_request_bytes = int(os.environ.get("ALTUR_MAX_REQUEST_BYTES") or 96 * 1024 * 1024)
        # 🔴 Modo de emergencia del runbook, y SOLO eso. Con el flag apagado (el default),
        # un bundle que no verifica deja el servicio en 503 en vez de responder una
        # constante: un 503 se ve en el monitor, una constante se lee como un modelo.
        self.emergency_constant = _env_flag("ALTUR_EMERGENCY_CONSTANT", False)
        # Petición ficticia al arrancar para absorber el cold start (planes de BLAS/FFT de
        # NumPy, primer toque de las páginas del intérprete). Ocurre ANTES de que readiness
        # responda 200, que es lo único que la hace útil: si el warm-up pasara después, el
        # primer request real de un juez seguiría pagándolo.
        self.warmup = _env_flag("ALTUR_WARMUP", True)


STATE: dict[str, Any] = {
    "detector": None, "problems": [], "settings": None, "loaded_at": None, "warmup_ms": None,
}


def _warmup(det: Any) -> tuple[float | None, str | None]:
    """Una inferencia ficticia sobre audio sintético. Nunca toca el dataset.

    Si el warm-up falla, el bundle no puede servir y el servicio NO se declara listo. Es
    deliberado: descubrirlo aquí cuesta un arranque; descubrirlo en el primer request de
    un juez cuesta la ronda.
    """
    import numpy as np

    from .types import SAMPLE_RATE, AudioExample

    rng = np.random.default_rng(0)   # fijo: el warm-up no debe introducir variabilidad
    n = SAMPLE_RATE * 2
    ruido = (rng.standard_normal(n) * 0.01).astype(np.float32)
    ex = AudioExample(ch0=ruido, ch1=ruido.copy(), sr=SAMPLE_RATE)
    t0 = time.perf_counter()
    try:
        det.predict(ex)
    except Exception as e:  # noqa: BLE001 — frontera: un warm-up roto es 503, no un crash
        return None, f"warm-up falló: {type(e).__name__}: {e}"
    return (time.perf_counter() - t0) * 1000.0, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    det, problems = bundle.load_detector(s.bundle_dir, strict=not s.emergency_constant)
    problems = list(problems)
    warmup_ms = None
    if det is not None and s.warmup:
        warmup_ms, warmup_problem = _warmup(det)
        if warmup_problem:
            problems.append(warmup_problem)
            det = None
    STATE.update(
        detector=det, problems=problems, settings=s, loaded_at=time.time(), warmup_ms=warmup_ms,
    )
    if det is None:
        # No se levanta una excepción: el proceso tiene que quedar vivo para que /health y
        # /version se puedan consultar y digan QUÉ falló. Un proceso muerto no diagnostica.
        log.error("bundle no servible; readiness quedará en 503: %s", problems)
    else:
        if problems:
            log.warning("bundle con observaciones: %s", problems)
        log.info("detector listo: %s (warm-up %s ms)", det.name, warmup_ms)
    yield
    STATE.update(detector=None)


app = FastAPI(
    title="altur-detect",
    version="0.1.0",
    description="¿La voz del canal 0 es humana o sintética?",
    lifespan=lifespan,
)


def _error(code: str, message: str, http: int) -> JSONResponse:
    """Errores sin stack trace y sin rutas. Solo un código estable y un mensaje."""
    return JSONResponse(status_code=http, content={"error": code, "detail": message})


async def _extract_audio_bytes(request: Request) -> bytes:
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    limit = STATE["settings"].max_request_bytes

    # multipart PRIMERO: `request.body()` consume el stream y dejaría `form()` vacío.
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        for key in (*AUDIO_FIELDS, *form.keys()):
            if key in form:
                v = form[key]
                data = await v.read() if hasattr(v, "read") else v
                if isinstance(data, bytes):
                    if len(data) > limit:
                        raise AudioDecodeError("archivo demasiado grande", "too_large")
                    return data
                return decode_base64(str(data))
        raise AudioDecodeError("multipart sin campo de audio", "missing_audio")

    raw = await request.body()
    if len(raw) > limit:
        raise AudioDecodeError("cuerpo de la petición demasiado grande", "too_large")
    if ctype in ("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"):
        return raw

    # Por defecto: JSON
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 — frontera: cualquier fallo de parseo es 4xx, no 500
        if raw[:4] == b"RIFF":          # alguien mandó WAV crudo sin content-type
            return raw
        raise AudioDecodeError("se esperaba un cuerpo JSON", "bad_json") from None
    if isinstance(payload, str):
        return decode_base64(payload)
    if not isinstance(payload, dict):
        raise AudioDecodeError("el JSON debe ser un objeto", "bad_json")
    for f in AUDIO_FIELDS:
        if f in payload and payload[f] is not None:
            return decode_base64(payload[f])
    raise AudioDecodeError(
        f"falta el campo de audio; se aceptan: {', '.join(AUDIO_FIELDS)}", "missing_audio"
    )


@app.post("/detect")
async def detect(request: Request) -> Response:
    s: Settings = STATE["settings"]
    det = STATE["detector"]
    if det is None:
        return _error("not_ready", "el detector no está cargado", status.HTTP_503_SERVICE_UNAVAILABLE)

    t0 = time.perf_counter()
    try:
        raw = await _extract_audio_bytes(request)
        decoded = decode(raw, allow_mono=s.allow_mono, allow_resample=s.allow_resample)
    except AudioDecodeError as e:
        return _error(e.code, str(e), status.HTTP_400_BAD_REQUEST)
    except Exception:
        log.exception("fallo inesperado decodificando")
        return _error("decode_failed", "no se pudo leer el audio", status.HTTP_400_BAD_REQUEST)

    try:
        pred: Prediction = det.predict(decoded.example)
    except Exception:
        log.exception("fallo inesperado en la inferencia")
        return _error("inference_failed", "fallo interno de inferencia",
                      status.HTTP_500_INTERNAL_SERVER_ERROR)

    body = pred.to_response(extras=s.response_extras)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "detect ok dur=%.1fs ch=%d sr=%d resampled=%s ms=%.1f",
        decoded.example.duration_s, decoded.original_channels,
        decoded.original_sr, decoded.was_resampled, elapsed_ms,
    )
    return JSONResponse(content=body, headers={"X-Inference-Ms": f"{elapsed_ms:.1f}"})


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness: el proceso responde. Siempre 200 si el servidor está de pie."""
    return {"status": "alive", "detector_loaded": STATE["detector"] is not None}


@app.get("/health/ready")
async def ready() -> Response:
    """Readiness: el modelo está cargado y puede atender. 503 si no."""
    if STATE["detector"] is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": STATE["problems"] or ["sin detector"]},
        )
    return JSONResponse(content={"status": "ready", "detector": STATE["detector"].name})


@app.get("/version")
async def version() -> dict[str, Any]:
    """Trazabilidad para los jueces. Sin rutas y sin secretos."""
    s: Settings = STATE["settings"]
    info: dict[str, Any] = {
        "service": "altur-detect",
        "version": app.version,
        "detector": STATE["detector"].name if STATE["detector"] else None,
        "git_commit": bundle.git_commit(),
        "git_dirty": bundle.git_dirty(),
        "response_extras": s.response_extras,
    }
    if s.bundle_dir:
        try:
            m = bundle.load_manifest(s.bundle_dir)
            info.update(
                bundle_created_utc=m.created_utc,
                bundle_commit=m.git_commit,
                protocol_id=m.protocol_id,
                run_id=m.run_id,
                n_features=len(m.feature_order),
                threshold=m.threshold,
                limitations=m.limitations,
            )
        except Exception:  # noqa: BLE001 — /version nunca debe tumbar el servicio
            info["bundle"] = "ilegible"
    if STATE["warmup_ms"] is not None:
        info["warmup_ms"] = round(float(STATE["warmup_ms"]), 1)
    if STATE["problems"]:
        # Se declara el fallo, no se esconde. Los mensajes de `bundle.verify` son nombres
        # relativos dentro del bundle (`model.json`), nunca rutas del host.
        key = "bundle_error" if STATE["detector"] is None else "warnings"
        info[key] = STATE["problems"]
        info["bundle_ok"] = STATE["detector"] is not None
    return info
