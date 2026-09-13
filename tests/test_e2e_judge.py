"""El runner e2e falla cerrado antes de tocar la red o `val`.

La prueba contra un servidor vivo solo corre con `ALTUR_E2E_URL` definida (y los datos
descargados): `ALTUR_E2E_URL=http://127.0.0.1:8000/detect make test`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e2e_judge", ROOT / "scripts" / "e2e_judge.py")
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)


def test_val_exige_use_val(capsys):
    assert e2e.main(["--url", "http://127.0.0.1:9/detect", "--split", "val"]) == 2
    assert "--use-val" in capsys.readouterr().err


def test_no_corre_con_un_cliente_que_no_es_el_oficial(tmp_path, monkeypatch):
    falso = tmp_path / "check_endpoint.py"
    falso.write_text("print('no soy el cliente del juez')\n")
    monkeypatch.setattr(e2e, "CHECKER", falso)
    assert e2e.main(["--url", "http://127.0.0.1:9/detect"]) == 2


@pytest.mark.parametrize(
    ("status", "raw", "ok"),
    [
        (200, b'{"is_synthetic": true, "confidence": 0.9}', True),
        (200, b'{"is_synthetic": false}', True),
        (200, b'{"is_synthetic": "true", "confidence": 0.9}', False),
        (200, b'{"is_synthetic": true, "confidence": 1.5}', False),
        (500, b'{"is_synthetic": true, "confidence": 0.9}', False),
    ],
)
def test_respuesta_valida_como_la_cuenta_el_juez(status, raw, ok):
    assert e2e.valid_answer(status, raw) is ok


def test_timeout_de_red_cuenta_como_llamada_fallida_y_no_aborta(monkeypatch):
    def lenta(*_a, **_k):
        raise e2e.urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(e2e.urllib.request, "urlopen", lenta)
    status, raw, _secs = e2e.request("http://127.0.0.1:9/detect", b"{}")
    assert status == 0
    assert not e2e.valid_answer(status, raw)


@pytest.mark.skipif(not os.environ.get("ALTUR_E2E_URL"), reason="define ALTUR_E2E_URL con un /detect vivo")
def test_endpoint_vivo_pasa_el_cliente_oficial(tmp_path):
    assert e2e.main(["--url", os.environ["ALTUR_E2E_URL"], "--n", "10", "--out-dir", str(tmp_path)]) == 0
