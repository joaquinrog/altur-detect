"""Criterios de aceptación de A4.1 y A4.2.

Lo que estos tests defienden no es que el bundle funcione, sino que **falle** en los cuatro
modos que producen un despliegue plausible y equivocado:

1. Un byte cambiado que nadie nota.
2. Un `feature_order` permutado, que da un número perfectamente válido y falso.
3. Una versión de schema que ya no es la que el cargador entiende.
4. Un bundle roto que se degrada en silencio a predicciones constantes.

El cuarto es el peor de todos, porque no deja rastro. Por eso la prueba no es "responde",
es "responde 503".
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from altur import bundle as bundle_mod
from altur.models.base import LinearExport
from altur.models.calibration import PlattCalibrator
from altur.registry import extractors
from altur.types import SAMPLE_RATE, AudioExample

FEATURES = tuple(
    f"a2.acoustic.ch0.{name}"
    for name in (
        "band_0_300_norm", "band_300_1000_norm", "band_1000_2000_norm",
        "band_2000_3000_norm", "band_3000_3400_norm", "band_3400_4001_norm",
        "spectral_flatness_log", "centroid_hz", "rolloff85_hz", "spectral_flux",
        "energy_log_iqr",
    )
)


def _export(order=FEATURES) -> LinearExport:
    n = len(order)
    rng = np.random.default_rng(3)
    return LinearExport(
        feature_order=tuple(order),
        mean=np.zeros(n),
        scale=np.ones(n),
        coef=rng.normal(size=n),
        intercept=0.1,
        name="test_linear@1",
    )


def _manifest(**kw) -> bundle_mod.BundleManifest:
    base = {
        "detector_name": "test_linear@1",
        "feature_order": list(FEATURES),
        "extractor_refs": ["a2.acoustic.ch0@1"],
        "segmenter_ref": "energy@1",
        "calibrator_ref": "platt@1",
        "threshold": 0.5,
    }
    base.update(kw)
    return bundle_mod.BundleManifest(**base)


def _calibrator() -> PlattCalibrator:
    c = PlattCalibrator()
    c.a_, c.b_ = 1.0, 0.0
    return c


@pytest.fixture()
def built(tmp_path: Path) -> Path:
    d = tmp_path / "bundle"
    bundle_mod.build_bundle(
        d, export=_export(), manifest=_manifest(), calibrator=_calibrator(),
        requirements="numpy==1.26.4\n",
    )
    return d


def _example(seconds: float = 3.0) -> AudioExample:
    rng = np.random.default_rng(11)
    n = int(seconds * SAMPLE_RATE)
    ch = (rng.standard_normal(n) * 0.05).astype(np.float32)
    return AudioExample(ch0=ch, ch1=ch.copy(), sr=SAMPLE_RATE)


# ------------------------------------------------------------------ A4.1 save-load-predict
def test_save_load_predict_dentro_de_tolerancia(built: Path):
    """El detector cargado del disco reproduce al export en memoria, bit a bit útil."""
    det, problems = bundle_mod.load_detector(built)
    assert problems == []
    ex = _example()
    p = det.predict(ex)

    esperado = _export().p_synthetic_from_mapping(det.features(ex))
    assert abs(p.diagnostics["p_synthetic"] - esperado) < 1e-6
    assert 0.0 <= p.confidence <= 1.0


def test_la_confianza_es_la_de_la_clase_reportada(built: Path):
    det, _ = bundle_mod.load_detector(built)
    p = det.predict(_example())
    esperada = p.diagnostics["p_synthetic"] if p.is_synthetic else 1 - p.diagnostics["p_synthetic"]
    assert abs(p.confidence - esperada) < 1e-9
    assert p.confidence >= 0.5, "la confianza en lo que se afirma nunca baja de 0.5"


def test_nunca_para_temprano(built: Path):
    """D-A3.8: `seconds_used` es siempre el audio completo, no un horizonte recortado."""
    det, _ = bundle_mod.load_detector(built)
    for seconds in (3.0, 12.0):
        p = det.predict(_example(seconds))
        assert p.seconds_used == pytest.approx(seconds, abs=1e-3)


# --------------------------------------------------------------------- A4.1 manipulaciones
def test_un_byte_cambiado_rompe_la_carga(built: Path):
    (built / "model.json").write_bytes((built / "model.json").read_bytes() + b" ")
    problems = bundle_mod.verify(built)
    assert any("hash distinto" in p for p in problems)
    det, problems = bundle_mod.load_detector(built)
    assert det is None, "un bundle manipulado NUNCA se sirve"


def test_orden_de_features_permutado_rompe_la_carga(built: Path):
    """Los hashes siguen en verde: lo que falla es la coherencia manifest/modelo."""
    m = json.loads((built / "manifest.json").read_text())
    m["feature_order"][0], m["feature_order"][1] = m["feature_order"][1], m["feature_order"][0]
    (built / "manifest.json").write_text(json.dumps(m))
    problems = bundle_mod.verify(built)
    assert any("feature_order" in p for p in problems)
    assert bundle_mod.load_detector(built)[0] is None


def test_version_de_schema_distinta_rompe_la_carga(built: Path):
    m = json.loads((built / "manifest.json").read_text())
    m["schema_version"] = bundle_mod.SCHEMA_VERSION + 1
    (built / "manifest.json").write_text(json.dumps(m))
    assert bundle_mod.verify(built)
    assert bundle_mod.load_detector(built)[0] is None


def test_archivo_no_declarado_rompe_la_carga(built: Path):
    (built / "colado.json").write_text("{}")
    assert any("no declarado" in p for p in bundle_mod.verify(built))
    assert bundle_mod.load_detector(built)[0] is None


# ------------------------------------------------------------------------ A4.1 privacidad
def test_el_manifest_no_lleva_anon_id(built: Path):
    texto = (built / "manifest.json").read_text()
    assert bundle_mod.scrub_problems(texto) == []


# Un id con la forma real (`call_` + hex) pero inventado. Se arma en tiempo de ejecución para
# que ningún literal con forma de anon_id viva en el código: el historial se audita con grep
# (D-A5.2) y un id real ya se coló una vez por un test (D-A4.6).
_FAKE_ANON_ID = "call_" + "0" * 12


@pytest.mark.parametrize(
    "payload,esperado",
    [
        (f'{{"nota": "{_FAKE_ANON_ID}"}}', "anon_id"),
        ('{"nota": "/home/joaquinrog/altur"}', "ruta local"),
        ('{"registry_token": "abc"}', "secreto"),
    ],
)
def test_el_scrub_caza_lo_que_no_se_publica(payload: str, esperado: str):
    problems = bundle_mod.scrub_problems(payload)
    assert any(esperado in p for p in problems), problems


def test_no_se_sella_un_manifest_impublicable(tmp_path: Path):
    d = tmp_path / "b"
    m = _manifest(run_id=_FAKE_ANON_ID)
    with pytest.raises(bundle_mod.BundleError, match="anon_id"):
        bundle_mod.build_bundle(d, export=_export(), manifest=m, calibrator=_calibrator())


# ------------------------------------------------------------------------- A4.1 licencias
def test_se_rechaza_un_extractor_no_apto_para_producto(tmp_path: Path):
    """Un extractor GPL-3 o research-only no se puede congelar en `/detect`."""

    @extractors.register(
        "test.gpl.solo", version=1, channels=(0,), needs_seg=False,
        license="GPL-3.0", product_safe=False,
    )
    def _extract(ex):  # pragma: no cover — nunca debe llegar a ejecutarse
        return {}, {}

    with pytest.raises(bundle_mod.BundleError, match="no aptos para producto"):
        bundle_mod.build_bundle(
            tmp_path / "b",
            export=_export(),
            manifest=_manifest(extractor_refs=["test.gpl.solo@1"]),
            calibrator=_calibrator(),
        )


def test_se_rechaza_una_referencia_flotante(tmp_path: Path):
    with pytest.raises(bundle_mod.BundleError, match="nombre@version"):
        bundle_mod.build_bundle(
            tmp_path / "b",
            export=_export(),
            manifest=_manifest(extractor_refs=["a2.acoustic.ch0"]),
            calibrator=_calibrator(),
        )


def test_se_rechaza_un_manifest_de_otro_experimento(tmp_path: Path):
    """El manifest dice 11 features y el export trae otras: no se sella."""
    with pytest.raises(bundle_mod.BundleError, match="no coincide"):
        bundle_mod.build_bundle(
            tmp_path / "b",
            export=_export(FEATURES[:5]),
            manifest=_manifest(),
            calibrator=_calibrator(),
        )


# -------------------------------------------------------------- A4.2 integración con /detect
def _client(bundle_dir: str | None, monkeypatch, **env):
    from fastapi.testclient import TestClient

    from altur.api import app

    monkeypatch.setenv("ALTUR_BUNDLE_DIR", bundle_dir or "")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(app)


def test_bundle_real_se_sirve_sin_tocar_api_py(built: Path, monkeypatch, golden_b64):
    """El bundle se carga POR CONFIGURACIÓN. `api.py` no sabe qué modelo es."""
    with _client(str(built), monkeypatch) as c:
        assert c.get("/health/ready").status_code == 200
        r = c.post("/detect", json={"audio": golden_b64})
        assert r.status_code == 200
        assert set(r.json()) == {"is_synthetic", "confidence"}
        assert c.get("/version").json()["detector"] == "test_linear@1"


def test_bundle_roto_deja_readiness_en_503(built: Path, monkeypatch, golden_b64):
    """🔴 El criterio central de A4.2: nunca una constante silenciosa."""
    (built / "model.json").write_bytes(b"{}")
    with _client(str(built), monkeypatch) as c:
        ready = c.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"

        # El proceso sigue vivo y DICE qué pasó, sin rutas del host.
        assert c.get("/health").status_code == 200
        version = c.get("/version").json()
        assert version["bundle_ok"] is False
        assert version["bundle_error"]
        assert "/home/" not in json.dumps(version)

        # Y sobre todo: no responde una predicción.
        assert c.post("/detect", json={"audio": golden_b64}).status_code == 503


def test_bundle_inexistente_deja_readiness_en_503(monkeypatch):
    with _client("models/no_existe_este_bundle", monkeypatch) as c:
        assert c.get("/health/ready").status_code == 503
        assert c.get("/version").json()["bundle_ok"] is False


def test_la_constante_solo_opera_por_modo_explicito(built: Path, monkeypatch, golden_b64):
    """`ConstantDetector` es el fallback del runbook, y solo con el flag encendido."""
    (built / "model.json").write_bytes(b"{}")
    with _client(str(built), monkeypatch, ALTUR_EMERGENCY_CONSTANT="1") as c:
        assert c.get("/health/ready").status_code == 200
        assert c.get("/version").json()["detector"] == "constant@1"
        assert c.post("/detect", json={"audio": golden_b64}).status_code == 200


def test_el_warmup_ocurre_antes_del_readiness(built: Path, monkeypatch):
    """El cold start lo paga el arranque, no el primer request de un juez."""
    with _client(str(built), monkeypatch) as c:
        assert c.get("/health/ready").status_code == 200
        assert c.get("/version").json()["warmup_ms"] >= 0


def test_sin_warmup_no_se_reporta(built: Path, monkeypatch):
    with _client(str(built), monkeypatch, ALTUR_WARMUP="0") as c:
        assert "warmup_ms" not in c.get("/version").json()


def test_determinismo_y_rafaga_con_bundle_real(built: Path, monkeypatch, golden_b64):
    """Dos workers no corrompen caché, modelo ni resultados (aceptación de A4.2)."""
    import concurrent.futures

    with _client(str(built), monkeypatch) as c:
        primera = c.post("/detect", json={"audio": golden_b64}).json()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            respuestas = [
                f.result()
                for f in [
                    pool.submit(c.post, "/detect", json={"audio": golden_b64})
                    for _ in range(16)
                ]
            ]
    assert all(r.status_code == 200 for r in respuestas)
    assert all(r.json() == primera for r in respuestas), "la ráfaga cambió el resultado"


def test_el_manifest_del_bundle_de_produccion_es_publicable():
    """El bundle real del repo, si existe, tiene que poder publicarse tal cual."""
    d = Path(__file__).resolve().parents[1] / "models" / "acoustic_ch0_v1"
    if not (d / "manifest.json").exists():
        pytest.skip("todavía no se ha corrido scripts/build_bundle.py")
    assert bundle_mod.verify(d) == []
    assert bundle_mod.scrub_problems((d / "manifest.json").read_text()) == []
    m = bundle_mod.load_manifest(d)
    assert m.limitations, "un bundle sin limitaciones declaradas no se publica"
    assert m.metrics["val_looks"] == 0
    assert m.protocol_id == "official_v1"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()
