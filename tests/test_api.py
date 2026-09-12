"""Tests de contrato de `/detect`.

Son el criterio de aceptación del endpoint. Si estos pasan, la ronda automatizada no se
pierde por un problema de formato — que es el modo de falla más barato de evitar y el más
caro de sufrir.
"""

from __future__ import annotations

import base64
import concurrent.futures

import pytest


# --------------------------------------------------------------------- contrato oficial
def test_contrato_exacto(client, golden_b64):
    r = client.post("/detect", json={"audio": golden_b64})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"is_synthetic", "confidence"}, "por defecto solo los campos oficiales"
    assert isinstance(body["is_synthetic"], bool)
    assert isinstance(body["confidence"], (int, float))
    assert 0.0 <= body["confidence"] <= 1.0


def test_reporta_latencia_en_cabecera(client, golden_b64):
    r = client.post("/detect", json={"audio": golden_b64})
    assert float(r.headers["X-Inference-Ms"]) >= 0


# ------------------------------------------------------------------ tolerancia de entrada
@pytest.mark.parametrize("campo", ["audio", "audio_base64", "audio_data", "wav", "data", "b64", "file"])
def test_alias_del_campo(client, golden_b64, campo):
    """El PDF no fija el nombre del campo (UNK). Aceptar alias es seguro barato."""
    assert client.post("/detect", json={campo: golden_b64}).status_code == 200


def test_json_que_es_solo_una_cadena(client, golden_b64):
    assert client.post("/detect", json=golden_b64).status_code == 200


def test_wav_crudo_en_el_cuerpo(client, golden_bytes):
    r = client.post("/detect", content=golden_bytes, headers={"content-type": "audio/wav"})
    assert r.status_code == 200


def test_wav_crudo_sin_content_type(client, golden_bytes):
    assert client.post("/detect", content=golden_bytes).status_code == 200


def test_multipart(client, golden_bytes):
    r = client.post("/detect", files={"audio": ("call.wav", golden_bytes, "audio/wav")})
    assert r.status_code == 200


def test_data_uri(client, golden_b64):
    r = client.post("/detect", json={"audio": "data:audio/wav;base64," + golden_b64})
    assert r.status_code == 200


# ------------------------------------------------------------------------ errores claros
@pytest.mark.parametrize(
    "payload,code",
    [
        ({}, "missing_audio"),
        ({"audio": ""}, "empty"),
        ({"audio": "!!!!no es base64!!!!"}, "bad_base64"),
        ({"audio": base64.b64encode(b"x" * 200).decode()}, "not_wav"),
        ({"audio": None}, "missing_audio"),
    ],
)
def test_errores_4xx_con_codigo(client, payload, code):
    r = client.post("/detect", json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == code


def test_los_errores_no_filtran_stack_trace(client):
    r = client.post("/detect", json={"audio": "!!!!"})
    cuerpo = r.text.lower()
    for fuga in ("traceback", "file \"", "/home/", "src/altur"):
        assert fuga not in cuerpo, f"la respuesta de error filtra {fuga!r}"


def test_json_malformado(client):
    r = client.post("/detect", content=b"{no es json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_json"


# ------------------------------------------------------------------------------ mono
def test_mono_se_degrada_explicitamente(client, mono_bytes):
    """Nunca se duplica ch0 en ch1. O se rechaza, o se declara la degradación."""
    r = client.post("/detect", content=mono_bytes, headers={"content-type": "audio/wav"})
    assert r.status_code in (200, 400)
    if r.status_code == 400:
        assert r.json()["error"] == "mono_not_allowed"
    else:
        assert set(r.json()) == {"is_synthetic", "confidence"}


# -------------------------------------------------------------------- salud y versión
def test_health_es_liveness(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_readiness_separado(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_version_trazable_y_sin_secretos(client):
    body = client.get("/version").json()
    assert body["service"] == "altur-detect"
    assert body["detector"]
    assert "git_commit" in body
    texto = str(body)
    for fuga in ("/home/", "API_KEY", "SECRET", ".venv"):
        assert fuga not in texto, f"/version filtra {fuga!r}"


# ------------------------------------------------------------------------ concurrencia
def test_rafaga_concurrente(client, golden_b64):
    """Los jueces corren SU benchmark: no puede serializarse ni romperse bajo ráfaga."""
    def hit(_):
        return client.post("/detect", json={"audio": golden_b64}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(hit, range(24))) == [200] * 24


def test_determinismo(client, golden_b64):
    a = client.post("/detect", json={"audio": golden_b64}).json()
    b = client.post("/detect", json={"audio": golden_b64}).json()
    assert a == b
