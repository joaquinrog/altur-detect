.DEFAULT_GOAL := help
PY      := ./.venv/bin/python
PIP     := ./.venv/bin/pip
PORT    ?= 8000
URL     ?= http://127.0.0.1:$(PORT)
IMAGE   ?= altur-detect:local
BUNDLE  ?= models/current
SPLIT   ?= train
N       ?= 40

.PHONY: help setup test lint serve smoke docker docker-run bundle bundle-lfcc bundle-test e2e \
	docs-pack clean data protocol leak-test failover-sim

help:  ## Muestra esta ayuda
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## venv + dependencias (inferencia + dev)
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "listo. 'make setup-train' añade el arnés de experimentos."

setup-train:  ## Dependencias del arnés de entrenamiento (NO van a la imagen)
	$(PIP) install -q -e ".[dev,train]"

data:  ## descarga el dataset oficial y verifica el SHA-256
	$(PY) scripts/download_data.py

protocol:  ## congela grupos y folds (determinista; corre despues de `make data`)
	$(PY) scripts/build_protocol.py

leak-test:  ## permuta etiquetas y verifica que el AUC cae a ~0.5
	$(PY) scripts/leak_test.py

test:  ## pytest
	$(PY) -m pytest

lint:  ## ruff
	$(PY) -m ruff check src tests scripts || true

# Sin ALTUR_BUNDLE_DIR el API arranca con ConstantDetector y readiness responde 200:
# es el arranque de A0, pero como fallback de demo seria una constante que se lee como
# un modelo. El bundle va explicito aqui, igual que en el Dockerfile.
BUNDLE_DIR ?= models/current

serve:  ## /detect en local sirviendo $(BUNDLE_DIR)
	ALTUR_BUNDLE_DIR=$(BUNDLE_DIR) ./.venv/bin/uvicorn altur.api:app --host 0.0.0.0 --port $(PORT) --workers 2

serve-dev:  ## /detect con recarga automática
	ALTUR_BUNDLE_DIR=$(BUNDLE_DIR) ./.venv/bin/uvicorn altur.api:app --port $(PORT) --reload

smoke:  ## Golpea un servidor VIVO: URL=http://host:puerto make smoke
	$(PY) scripts/smoke.py --url $(URL)

docker:  ## Construye la imagen
	docker build -t $(IMAGE) .

docker-run:  ## Corre la imagen en $(PORT)
	docker run --rm -p $(PORT):8000 --name altur-detect $(IMAGE)

bundle:  ## Entrena el candidato y sella el bundle en models/acoustic_ch0_v1
	$(PY) scripts/build_bundle.py --out models/acoustic_ch0_v1
	@echo "  selecciona el candidato con: cp -a models/acoustic_ch0_v1 models/current"

bundle-lfcc:  ## C2 de D-A7.3 (LFCC portado a NumPy) en models/spectral_factory_lfcc_v1
	$(PY) scripts/build_bundle_lfcc.py --out models/spectral_factory_lfcc_v1

failover-sim:  ## Ejecuta las 4 simulaciones locales del runbook (A4.4)
	$(PY) scripts/build_bundle.py --out /tmp/altur-prev >/dev/null
	$(PY) scripts/failover_sim.py --bundle models/current --previous /tmp/altur-prev

bundle-test:  ## 🔴 Construir, guardar, cargar y predecir en un contenedor LIMPIO
	docker build -t $(IMAGE) .
	docker run --rm -d -p 18000:8000 --name altur-bundle-test $(IMAGE)
	@sleep 6
	-$(PY) scripts/smoke.py --url http://127.0.0.1:18000
	@docker rm -f altur-bundle-test >/dev/null

e2e:  ## 🔴 Cliente oficial del juez vs contenedor limpio (2 CPU, 8 GB): BUNDLE=... SPLIT=train N=40
	docker build -t $(IMAGE) .
	$(PY) scripts/e2e_judge.py --image $(IMAGE) --bundle $(BUNDLE) --split $(SPLIT) --n $(N)

docs-pack:  ## Empaqueta los docs internos (no versionados) para pasarlos al equipo
	@tar czf /tmp/altur-docs.tar.gz AGENTS.md CLAUDE.md docs/ 2>/dev/null && \
	 echo "  /tmp/altur-docs.tar.gz  ->  desempaqueta con: tar xzf altur-docs.tar.gz -C <repo>" || \
	 echo "  nada que empaquetar"

clean:  ## Borra cachés de Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
