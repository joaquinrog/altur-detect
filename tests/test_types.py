import numpy as np
import pytest

from altur.types import AudioExample, DatasetRecord, Prediction, Segmentation, Turn


def test_arrays_read_only():
    """`frozen=True` NO congela un ndarray. Esto sí."""
    ex = AudioExample(ch0=np.zeros(8000), ch1=np.zeros(8000))
    with pytest.raises(ValueError):
        ex.ch0[0] = 1.0


def test_canales_desalineados_fallan():
    with pytest.raises(ValueError):
        AudioExample(ch0=np.zeros(8000), ch1=np.zeros(4000))


def test_sample_rate_estricto():
    with pytest.raises(ValueError):
        AudioExample(ch0=np.zeros(8000), ch1=None, sr=16000)


def test_el_extractor_no_puede_ver_metadatos():
    """El candado: los campos sensibles no existen en AudioExample."""
    ex = AudioExample(ch0=np.zeros(8000), ch1=np.zeros(8000))
    for prohibido in ("example_id", "label", "split", "groups", "provenance", "call_id"):
        assert not hasattr(ex, prohibido), f"AudioExample expone {prohibido}"


def test_dataset_record_separado():
    rec = DatasetRecord(example_id="call_x", label=1, split="train",
                        groups={"speaker_or_voice_id": "s3"})
    assert rec.label == 1
    assert not rec.is_derived
    assert DatasetRecord("y", provenance={"transforms": ["mulaw"]}).is_derived


def test_speech_mask():
    seg = Segmentation(turns=(Turn(0, 1.0, 2.0), Turn(1, 3.0, 4.0)), source="webrtc@1")
    m = seg.speech_mask(0, 8000 * 5)
    assert m.sum() == 8000
    assert m[:8000].sum() == 0
    assert seg.speech_mask(1, 8000 * 5).sum() == 8000


def test_oracle_se_declara():
    assert Segmentation((), "oracle@1").is_oracle
    assert not Segmentation((), "webrtc@1").is_oracle


def test_respuesta_por_defecto_solo_contrato_oficial():
    r = Prediction(True, 0.87, seconds_used=12.0, degraded="ch0_only").to_response(extras=False)
    assert set(r) == {"is_synthetic", "confidence"}
    assert r["is_synthetic"] is True


def test_respuesta_con_extras():
    r = Prediction(False, 0.2, seconds_used=12.0).to_response(extras=True)
    assert r["seconds_used"] == 12.0
    assert "degraded" not in r


def test_confianza_fuera_de_rango():
    from altur.detector import ConstantDetector
    with pytest.raises(ValueError):
        ConstantDetector(confidence=1.5)


def test_el_wav_dorado_es_determinista(tmp_path):
    """El fixture se genera, no se commitea. Por eso su determinismo es un contrato."""
    import hashlib

    from tests.conftest import _synth
    from altur.io import write_wav

    hashes = []
    for i in range(2):
        p = tmp_path / f"g{i}.wav"
        ch0, ch1 = _synth()
        write_wav(str(p), ch0, ch1)
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1], "el fixture dorado debe ser byte-idéntico entre corridas"
