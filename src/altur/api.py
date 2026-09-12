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


STATE: dict[str, Any] = {"detector": None, "problems": [], "settings": None, "loaded_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    det, problems = bundle.load_detector(s.bundle_dir)
    STATE.update(detector=det, problems=problems, settings=s, loaded_at=time.time())
    if problems:
        log.warning("bundle con observaciones: %s", problems)
    log.info("detector listo: %s", det.name)
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
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"status": "not_ready"})
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
    if STATE["problems"]:
        info["warnings"] = STATE["problems"]
    return info
