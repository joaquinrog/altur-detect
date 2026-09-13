"""Monitor de la mesa: base commiteada en la sesión 13, apagada por defecto.

Lo que estos tests fijan es lo que no puede romperse mientras se diseña el resto:
el contrato de `/detect` intacto, el monitor invisible si nadie lo enciende, y un
fallo del monitor que no arrastre al endpoint.
"""

from __future__ import annotations

import pytest

from altur import monitor


@pytest.fixture()
def monitor_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ALTUR_MONITOR", "1")
    monkeypatch.setenv("ALTUR_MONITOR_PATH", str(tmp_path / "calls.jsonl"))
    from fastapi.testclient import TestClient

    from altur.api import app
    with TestClient(app) as c:
        yield c


def test_monitor_apagado_no_existe(client):
    """Sin la variable, las dos rutas responden 404: no se anuncia lo que no está activo."""
    assert client.get("/monitor").status_code == 404
    assert client.get("/monitor/calls").status_code == 404


def test_detect_no_cambia_con_el_monitor_encendido(monitor_client, golden_b64):
    """El contrato manda: encender el monitor no añade ni un campo a la respuesta."""
    r = monitor_client.post("/detect", json={"call_id": "call_X", "audio_base64": golden_b64})
    assert r.status_code == 200
    assert set(r.json()) == {"is_synthetic", "confidence"}


def test_monitor_registra_la_llamada_sin_guardar_el_call_id(monitor_client, golden_b64):
    monitor_client.post("/detect", json={"call_id": "call_SECRETO", "audio_base64": golden_b64})
    data = monitor_client.get("/monitor/calls").json()

    assert data["ready"] is True
    assert data["summary"]["calls"] == 1
    call = data["calls"][0]
    assert call["status"] == 200
    assert isinstance(call["is_synthetic"], bool)
    assert 0.0 <= call["confidence"] <= 1.0
    assert call["ms"] >= 0
    assert "call_SECRETO" not in str(data), "el call_id no se guarda, solo una referencia derivada"
    assert call["ref"] == monitor.call_ref("call_SECRETO")


def test_monitor_registra_los_errores(monitor_client):
    monitor_client.post("/detect", content=b"{no es json", headers={"Content-Type": "application/json"})
    data = monitor_client.get("/monitor/calls").json()
    assert data["summary"]["errors"] == 1
    assert data["calls"][0]["status"] == 400


def test_la_pagina_se_sirve_sin_dependencias_externas(monitor_client):
    """El contenedor no tiene salida a internet: un CDN dejaría la página en blanco."""
    html = monitor_client.get("/monitor").text
    assert "<title>altur-detect · monitor</title>" in html
    assert "http://" not in html and "https://" not in html


def test_un_monitor_roto_no_tumba_detect(monitor_client, golden_b64, monkeypatch):
    """La regla nº 1 de monitor.py, ejercida: si el registro falla, /detect ni se entera."""
    def boom(*_a, **_k):
        raise OSError("disco lleno")

    monkeypatch.setattr(monitor.Path, "open", boom)
    r = monitor_client.post("/detect", json={"audio_base64": golden_b64})
    assert r.status_code == 200
    assert set(r.json()) == {"is_synthetic", "confidence"}


def test_lectura_tolera_lineas_a_medias(tmp_path, monkeypatch):
    """Con varios workers escribiendo, una línea truncada no puede vaciar el monitor."""
    p = tmp_path / "calls.jsonl"
    p.write_text('{"ts": 1, "status": 200, "ms": 5, "is_synthetic": true}\n{"ts": 2, "stat\n',
                 encoding="utf-8")
    monkeypatch.setenv("ALTUR_MONITOR_PATH", str(p))
    calls = monitor.read()
    assert len(calls) == 1
    assert monitor.summary(calls)["synthetic"] == 1
