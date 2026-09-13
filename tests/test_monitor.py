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


# --- el registro fuera del camino de la respuesta (D-A7.9) -------------------------------

def test_watch_debe_seguir_siendo_sync():
    """🔴 `_watch` va en un BackgroundTask y TIENE que ser síncrona.

    Starlette manda al threadpool solo los callables no-corrutina (`background.py`:
    `is_async_callable` → `run_in_threadpool`). Si alguien la convierte a `async def`,
    pasa a correr en el event loop y su I/O bloquea al worker — sin que falle nada
    visible. Este test es la única defensa contra esa regresión.
    """
    from starlette._utils import is_async_callable

    from altur import api
    assert not is_async_callable(api._watch)


def test_el_registro_no_entra_en_la_respuesta(monitor_client, golden_b64):
    """El cuerpo se manda antes de correr la tarea, así que la cabecera no la incluye."""
    r = monitor_client.post("/detect", json={"call_id": "c", "audio_base64": golden_b64})
    assert r.status_code == 200
    assert float(r.headers["X-Inference-Ms"]) >= 0
    # Y aun así quedó registrada: BackgroundTask corre, solo que después.
    assert monitor_client.get("/monitor/calls").json()["summary"]["calls"] == 1


def test_upload_ms_va_en_la_misma_linea_que_el_resto(monitor_client, golden_b64):
    """Un registro por llamada: subida y modelo atados por `ref`, no dos números sueltos."""
    monitor_client.post("/detect", json={"call_id": "call_U", "audio_base64": golden_b64})
    call = monitor_client.get("/monitor/calls").json()["calls"][0]

    assert call["ref"] == monitor.call_ref("call_U")
    assert isinstance(call["upload_ms"], (int, float)) and call["upload_ms"] >= 0
    assert isinstance(call["ms"], (int, float))
    # El cronómetro del handler completo tiene que cubrir a los otros dos.
    assert call["server_ms"] >= call["upload_ms"] + call["ms"]


def test_upload_ms_no_toca_el_contrato(monitor_client, golden_b64):
    """La subida es del monitor. Ni cabecera nueva ni campo nuevo en el cuerpo."""
    r = monitor_client.post("/detect", json={"audio_base64": golden_b64})
    assert set(r.json()) == {"is_synthetic", "confidence"}
    assert not [h for h in r.headers if "upload" in h.lower()]


def test_sin_upload_ms_el_registro_no_revienta(monitor_client):
    """En el 503 de not_ready, `_watch` corre antes de que el cuerpo se haya leído."""
    from altur.api import STATE

    det, STATE["detector"] = STATE["detector"], None
    try:
        assert monitor_client.post("/detect", json={"audio_base64": "x"}).status_code == 503
    finally:
        STATE["detector"] = det
    call = monitor_client.get("/monitor/calls").json()["calls"][0]
    assert call["status"] == 503 and call["upload_ms"] is None


# --- el presupuesto de 30 s (el número que mira el juez) ---------------------------------

def test_el_total_es_el_trabajo_entero_del_servidor_no_subida_mas_modelo():
    """🔴 Medido: `upload_ms + ms` se queda a la mitad.

    Entre la subida y la inferencia está el parseo del JSON, el base64 y el WAV, que no
    entran en ninguno de los dos. Con el cliente oficial sobre localhost, sumar solo esos
    dos daba 58 ms contra los 139 ms que cronometró el cliente.
    """
    c = {"server_ms": 139.0, "upload_ms": 11.5, "ms": 45.6}
    assert monitor.total_ms(c) == 139.0
    assert monitor.decode_ms(c) == round(139.0 - 11.5 - 45.6, 10)


def test_las_tres_etapas_suman_el_total():
    c = {"server_ms": 100.0, "upload_ms": 20.0, "ms": 30.0}
    assert monitor.decode_ms(c) == 50.0
    assert c["upload_ms"] + monitor.decode_ms(c) + c["ms"] == monitor.total_ms(c)


def test_total_cae_a_la_suma_si_no_hay_server_ms():
    """Registros viejos o rutas de error sin el cronómetro completo."""
    assert monitor.total_ms({"upload_ms": 1680.0, "ms": 140.0}) == 1820.0
    assert monitor.total_ms({"ms": 140.0}) == 140.0
    assert monitor.total_ms({"status": 503}) is None
    assert monitor.decode_ms({"upload_ms": 1.0, "ms": 2.0}) is None


def test_decode_nunca_sale_negativo():
    """Los tres cronómetros no arrancan en el mismo instante; el redondeo puede cruzarlos."""
    assert monitor.decode_ms({"server_ms": 50.0, "upload_ms": 30.0, "ms": 25.0}) == 0.0


def test_el_margen_sale_de_la_peor_llamada_no_del_promedio():
    """El juez cuenta como fallo cada llamada que se pase, así que manda la peor."""
    s = monitor.summary([
        {"status": 200, "server_ms": 1840.0, "upload_ms": 1700.0, "ms": 140.0},
        {"status": 200, "server_ms": 300.0, "upload_ms": 200.0, "ms": 100.0},
    ])
    assert s["budget_s"] == 30.0
    assert s["total_ms"]["max"] == 1840.0
    assert s["worst_headroom_x"] == round(30_000 / 1840.0, 1)   # ~16.3x


def test_el_desglose_es_de_una_sola_llamada_y_suma_el_titular():
    """🔴 Los máximos por etapa salen de llamadas distintas y NO suman el total.

    La cabecera dice "peor llamada contra el presupuesto", así que el desglose tiene que
    ser el de esa llamada. Aquí la peor por total es la primera, aunque la segunda tenga
    el modelo más lento y la tercera la subida más lenta.
    """
    s = monitor.summary([
        {"status": 200, "server_ms": 500.0, "upload_ms": 100.0, "ms": 50.0},   # la peor
        {"status": 200, "server_ms": 400.0, "upload_ms": 10.0, "ms": 300.0},   # modelo peor
        {"status": 200, "server_ms": 300.0, "upload_ms": 250.0, "ms": 20.0},   # subida peor
    ])
    w = s["worst"]
    assert w["total_ms"] == 500.0
    assert (w["upload_ms"], w["inference_ms"]) == (100.0, 50.0)
    assert w["upload_ms"] + w["decode_ms"] + w["inference_ms"] == w["total_ms"]
    # Y los máximos por etapa siguen existiendo, pero son otra cosa:
    assert s["inference_ms"]["max"] == 300.0 and s["upload_ms"]["max"] == 250.0


def test_summary_vacio_no_revienta():
    s = monitor.summary([])
    assert s["calls"] == 0
    assert s["total_ms"] == {"p50": None, "p95": None, "max": None}
    assert s["decode_ms"] == {"p50": None, "p95": None, "max": None}
    assert s["worst_headroom_x"] is None
    assert s["worst"] is None


# --- el token de las rutas del monitor ---------------------------------------------------

@pytest.fixture()
def monitor_token_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ALTUR_MONITOR", "1")
    monkeypatch.setenv("ALTUR_MONITOR_PATH", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("ALTUR_MONITOR_TOKEN", "s3creto")
    from fastapi.testclient import TestClient

    from altur.api import app
    with TestClient(app) as c:
        yield c


def test_sin_token_el_monitor_no_existe(monitor_token_client):
    """404 y no 401: un 401 confirmaría que hay algo detrás."""
    assert monitor_token_client.get("/monitor").status_code == 404
    assert monitor_token_client.get("/monitor/calls").status_code == 404
    assert monitor_token_client.get("/monitor?k=otro").status_code == 404


def test_con_token_se_abre_por_query_o_por_cabecera(monitor_token_client):
    assert monitor_token_client.get("/monitor?k=s3creto").status_code == 200
    assert monitor_token_client.get(
        "/monitor/calls", headers={"X-Altur-Monitor-Token": "s3creto"},
    ).status_code == 200


def test_el_token_no_resucita_un_monitor_apagado(client):
    """La bandera manda: con ALTUR_MONITOR apagado, ningún token abre nada."""
    assert client.get("/monitor?k=s3creto").status_code == 404
