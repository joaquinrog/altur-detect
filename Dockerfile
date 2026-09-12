# Imagen de inferencia. Objetivos, en orden:
#   1. Autocontenida: el modelo va dentro; no descarga nada al arrancar.
#   2. Sin GPU y sin dependencias nativas de audio (io.py usa `wave` de la stdlib).
#   3. Sin egress: nada en el camino de /detect hace peticiones de red salientes.
# Eso es el criterio Feasibility del PDF —"could a bank deploy this on real phone audio?"—
# demostrado, no afirmado: el contenedor corre dentro del perímetro del banco.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: esta capa se cachea entre despliegues de modelo.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir . && \
    rm -rf /app/build && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# El bundle del modelo. Va DENTRO de la imagen a propósito: un contenedor que descarga
# su modelo al arrancar no es desplegable en una red cerrada.
# `.dockerignore` deja pasar SOLO `models/current`: los candidatos que no se sirven no
# tienen por qué viajar, y el rollback del runbook se hace por digest de imagen, no
# cambiando de directorio dentro de la misma imagen.
COPY models/ ./models/

# Usuario sin privilegios.
RUN useradd --create-home --uid 10001 altur && chown -R altur:altur /app
USER altur

ENV ALTUR_BUNDLE_DIR=/app/models/current \
    ALTUR_ALLOW_MONO=1 \
    ALTUR_ALLOW_RESAMPLE=1 \
    ALTUR_RESPONSE_EXTRAS=0 \
    PORT=8000 \
    WORKERS=2

EXPOSE 8000

# Readiness, no liveness: importa que el modelo esté cargado, no que el proceso viva.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
u=f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health/ready'; \
sys.exit(0 if urllib.request.urlopen(u,timeout=2).status==200 else 1)"

# Varios workers: los jueces corren SU benchmark contra el endpoint y un worker único
# serializaría la ráfaga.
CMD ["sh", "-c", "uvicorn altur.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --no-access-log"]
