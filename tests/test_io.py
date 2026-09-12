import base64

import numpy as np
import pytest

from altur.io import AudioDecodeError, decode, decode_base64, decode_payload
from altur.types import SAMPLE_RATE


def test_estereo_ok(golden_bytes):
    d = decode(golden_bytes)
    assert d.example.sr == SAMPLE_RATE
    assert not d.example.is_mono
    assert d.original_channels == 2
    assert not d.was_resampled
    assert d.example_id.startswith("sha256:")


def test_hash_es_estable(golden_bytes):
    assert decode(golden_bytes).sha256 == decode(golden_bytes).sha256


def test_mono_no_se_duplica(mono_bytes):
    """La regla que impide inventar un canal de agente."""
    d = decode(mono_bytes, allow_mono=True)
    assert d.example.is_mono
    assert d.example.ch1 is None
    with pytest.raises(ValueError):
        d.example.channel(1)


def test_mono_puede_rechazarse(mono_bytes):
    with pytest.raises(AudioDecodeError) as e:
        decode(mono_bytes, allow_mono=False)
    assert e.value.code == "mono_not_allowed"


def test_remuestreo_defensivo(wideband_bytes):
    d = decode(wideband_bytes, allow_resample=True)
    assert d.example.sr == SAMPLE_RATE
    assert d.was_resampled
    assert d.original_sr == 16000
    assert abs(d.example.duration_s - 4.0) < 0.01


def test_remuestreo_puede_rechazarse(wideband_bytes):
    with pytest.raises(AudioDecodeError) as e:
        decode(wideband_bytes, allow_resample=False)
    assert e.value.code == "bad_sr"


def test_data_uri(golden_b64):
    assert decode_payload("data:audio/wav;base64," + golden_b64).example.duration_s > 0


def test_base64_con_espacios(golden_b64):
    troceado = "\n".join(golden_b64[i:i + 76] for i in range(0, len(golden_b64), 76))
    assert decode_payload(troceado).example.duration_s > 0


@pytest.mark.parametrize(
    "payload,code",
    [("", "empty"), ("!!!!", "bad_base64"), (base64.b64encode(b"x" * 100).decode(), "not_wav")],
)
def test_entradas_invalidas(payload, code):
    with pytest.raises(AudioDecodeError) as e:
        decode_payload(payload)
    assert e.value.code == code


def test_demasiado_corto(tmp_path):
    from altur.io import write_wav
    p = tmp_path / "corto.wav"
    write_wav(str(p), np.zeros(100), np.zeros(100))
    with pytest.raises(AudioDecodeError) as e:
        decode(p.read_bytes())
    assert e.value.code == "too_short"


def test_no_es_string():
    with pytest.raises(AudioDecodeError) as e:
        decode_base64(12345)  # type: ignore[arg-type]
    assert e.value.code == "not_a_string"
