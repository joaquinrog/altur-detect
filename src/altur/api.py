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
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from . import bundle, monitor
from .io import AudioDecodeError, decode, decode_base64
from .monitor_ui import PAGE as MONITOR_PAGE
from .types import Prediction

log = logging.getLogger("altur.api")

# `audio_base64` es el campo oficial. Los alias se conservan como tolerancia secundaria;
# el smoke y los tests de aceptación usan siempre el contrato exacto del juez.
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
        # Monitor de la mesa: registro de llamadas y página de estado. Apagado por defecto,
        # y cuando está encendido no toca la respuesta de /detect (ver monitor.py).
        self.monitor = _env_flag("ALTUR_MONITOR", False)
        # Token de las rutas del monitor. El endpoint es público y está en la misma IP:puerto
        # que /detect: sin token, cualquiera con la URL ve el tráfico del turno de juicio.
        # Vacío = sin puerta (solo para desarrollo local).
        self.monitor_token = (os.environ.get("ALTUR_MONITOR_TOKEN") or "").strip()


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

    # Cronómetro de la subida, para el monitor. Va justo aquí y no más arriba porque esta
    # línea es la única que espera a que el cliente termine de mandar el cuerpo: el parseo
    # JSON y el base64 vienen después y son trabajo nuestro, no de la red.
    # NO entra en X-Inference-Ms (D-A6.6: la cabecera mide el modelo) ni en la respuesta.
    t_up = time.perf_counter()
    raw = await request.body()
    request.state.upload_ms = (time.perf_counter() - t_up) * 1000.0
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
    # Para el monitor: se guarda una referencia derivada, nunca el `call_id` en claro.
    request.state.call_ref = monitor.call_ref(payload.get("call_id"))
    for f in AUDIO_FIELDS:
        if f in payload and payload[f] is not None:
            return decode_base64(payload[f])
    raise AudioDecodeError(
        f"falta el campo de audio; se aceptan: {', '.join(AUDIO_FIELDS)}", "missing_audio"
    )


def _watch(request: Request, s: Settings, **fields: Any) -> None:
    """Anota la llamada si el monitor está encendido. Ver monitor.py: nunca lanza.

    🔴 TIENE QUE SEGUIR SIENDO `def`, no `async def`. Se pasa a `BackgroundTask`, y
    Starlette manda al threadpool solo los callables síncronos (`background.py`:
    `is_async_callable` → `run_in_threadpool`). Convertirla a corrutina la movería al
    event loop y su I/O bloquearía al worker, en silencio. Lo fija
    `test_watch_debe_seguir_siendo_sync`.
    """
    if not s.monitor:
        return
    try:
        size = int(request.headers.get("content-length") or 0)
    except ValueError:
        size = 0
    up = getattr(request.state, "upload_ms", None)   # no existe en el 503 de not_ready
    monitor.record(
        ref=getattr(request.state, "call_ref", None),
        mb=round(size / 1_048_576, 3) if size else None,
        upload_ms=round(up, 1) if isinstance(up, float) else None,
        **fields,
    )


def _monitor_open(request: Request, s: Settings) -> bool:
    """¿Se puede servir una ruta del monitor? Apagado o token malo => no existe.

    `compare_digest` sobre bytes, no `==`: la comparación normal sale antes en el primer
    byte distinto. Aquí el riesgo real es bajo, pero hacerlo bien es gratis.
    """
    if not s.monitor:
        return False
    if not s.monitor_token:
        return True
    given = request.query_params.get("k") or request.headers.get("x-altur-monitor-token") or ""
    return secrets.compare_digest(given.encode("utf-8"), s.monitor_token.encode("utf-8"))


@app.post("/detect")
async def detect(request: Request) -> Response:
    # Cronómetro de TODO lo que hace el servidor: subida, parseo/decodificación e inferencia.
    # Hace falta aparte de `upload_ms` y de `X-Inference-Ms` porque entre los dos queda un
    # hueco que ninguno cubre —parsear ~6 MB de JSON, el base64 y el WAV— y medido son
    # decenas de ms. Sin esto, el "de 30 s" de la página se quedaría corto.
    t_req = time.perf_counter()
    s: Settings = STATE["settings"]
    det = STATE["detector"]
    if det is None:
        _watch(request, s, status=503, error="not_ready")
        return _error("not_ready", "el detector no está cargado", status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        raw = await _extract_audio_bytes(request)
        decoded = decode(raw, allow_mono=s.allow_mono, allow_resample=s.allow_resample)
    except AudioDecodeError as e:
        _watch(request, s, status=400, error=e.code)
        return _error(e.code, str(e), status.HTTP_400_BAD_REQUEST)
    except Exception:
        log.exception("fallo inesperado decodificando")
        _watch(request, s, status=400, error="decode_failed")
        return _error("decode_failed", "no se pudo leer el audio", status.HTTP_400_BAD_REQUEST)

    # La cabecera mide el trabajo del modelo, no la subida ni el parseo del JSON.
    t0 = time.perf_counter()
    try:
        pred: Prediction = det.predict(decoded.example)
    except Exception:
        log.exception("fallo inesperado en la inferencia")
        _watch(request, s, status=500, error="inference_failed",
               ms=(time.perf_counter() - t0) * 1000.0)
        return _error("inference_failed", "fallo interno de inferencia",
                      status.HTTP_500_INTERNAL_SERVER_ERROR)

    body = pred.to_response(extras=s.response_extras)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # El registro va como BackgroundTask: Starlette manda el cuerpo y solo DESPUÉS corre la
    # tarea (responses.py: `await send(http.response.body)` precede a `await background()`),
    # así que no entra en la latencia que mide el cliente del juez. Y como `_watch` es
    # síncrona, corre en el threadpool y no bloquea el event loop.
    # No es por ahorrar microsegundos —el append cuesta ~50 µs— sino por `_truncate`, que
    # lee 1 MB y lo reescribe cuando el JSONL se pasa del tope.
    watch = BackgroundTask(
        _watch, request, s, status=200, ms=round(elapsed_ms, 2),
        server_ms=round((time.perf_counter() - t_req) * 1000.0, 1),
        is_synthetic=bool(pred.is_synthetic), confidence=round(float(pred.confidence), 4),
        duration_s=round(decoded.example.duration_s, 1),
    )
    log.info(
        "detect ok dur=%.1fs ch=%d sr=%d resampled=%s ms=%.1f",
        decoded.example.duration_s, decoded.original_channels,
        decoded.original_sr, decoded.was_resampled, elapsed_ms,
    )
    return JSONResponse(
        content=body, headers={"X-Inference-Ms": f"{elapsed_ms:.1f}"}, background=watch,
    )


@app.get("/monitor", include_in_schema=False)
async def monitor_page(request: Request) -> Response:
    """La página de la mesa. 404 si está apagado o el token no cuadra.

    404 y no 401 a propósito: un 401 confirmaría que la ruta existe y que hay algo detrás.
    """
    s: Settings = STATE["settings"]
    if not _monitor_open(request, s):
        return _error("not_found", "monitor desactivado", status.HTTP_404_NOT_FOUND)
    return Response(content=MONITOR_PAGE, media_type="text/html; charset=utf-8")


@app.get("/monitor/calls", include_in_schema=False)
async def monitor_calls(request: Request, limit: int = monitor.DEFAULT_LIMIT) -> Response:
    """Lo que alimenta la página: últimas llamadas y agregados. Nunca toca el detector."""
    s: Settings = STATE["settings"]
    if not _monitor_open(request, s):
        return _error("not_found", "monitor desactivado", status.HTTP_404_NOT_FOUND)
    det = STATE["detector"]
    calls = monitor.read(max(1, min(int(limit), 500)))
    info: dict[str, Any] = {
        "ready": det is not None,
        "detector": det.name if det else None,
        "calls": calls,
        "summary": monitor.summary(calls),
    }
    if s.bundle_dir:
        try:
            info["threshold"] = bundle.load_manifest(s.bundle_dir).threshold
        except Exception:  # el monitor nunca tumba nada
            log.debug("monitor: manifest ilegible", exc_info=True)
    return JSONResponse(content=info)


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
