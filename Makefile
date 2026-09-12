.DEFAULT_GOAL := help
PY      := ./.venv/bin/python
PIP     := ./.venv/bin/pip
PORT    ?= 8000
URL     ?= http://127.0.0.1:$(PORT)
IMAGE   ?= altur-detect:local

.PHONY: help setup test lint serve smoke docker docker-run bundle-test docs-pack clean \
	data protocol leak-test

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

serve:  ## /detect en local
	./.venv/bin/uvicorn altur.api:app --host 0.0.0.0 --port $(PORT) --workers 2

serve-dev:  ## /detect con recarga automática
	./.venv/bin/uvicorn altur.api:app --port $(PORT) --reload

smoke:  ## Golpea un servidor VIVO: URL=http://host:puerto make smoke
	$(PY) scripts/smoke.py --url $(URL)

docker:  ## Construye la imagen
	docker build -t $(IMAGE) .

docker-run:  ## Corre la imagen en $(PORT)
	docker run --rm -p $(PORT):8000 --name altur-detect $(IMAGE)

bundle-test:  ## 🔴 Construir, guardar, cargar y predecir en un contenedor LIMPIO
	docker build -t $(IMAGE) .
	docker run --rm -d -p 18000:8000 --name altur-bundle-test $(IMAGE)
	@sleep 6
	-$(PY) scripts/smoke.py --url http://127.0.0.1:18000
	@docker rm -f altur-bundle-test >/dev/null

docs-pack:  ## Empaqueta los docs internos (no versionados) para pasarlos al equipo
	@tar czf /tmp/altur-docs.tar.gz AGENTS.md CLAUDE.md docs/ 2>/dev/null && \
	 echo "  /tmp/altur-docs.tar.gz  ->  desempaqueta con: tar xzf altur-docs.tar.gz -C <repo>" || \
	 echo "  nada que empaquetar"

clean:  ## Borra cachés de Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
