import numpy as np

from altur.features.acoustic_minimal import (
    FEATURE_ORDER,
    SILENCE_FEATURE_ORDER,
    SPEECH_FEATURE_ORDER,
    _frames,
    extract,
    extract_ch1,
    extract_silence,
    extract_speech,
)
from altur.types import AudioExample, Segmentation, Turn


def _example(seconds=1.0, *, ch1=False):
    n = int(seconds * 8000)
    t = np.arange(n, dtype=np.float32) / 8000
    ch0 = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    other = (0.2 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32) if ch1 else None
    return AudioExample(ch0, other)


def test_schema_registrado_ordenado_y_namespaceado():
    from altur.registry import extractors

    assert extractors.resolve("a2.acoustic.ch0@1") is extract
    assert extractors.resolve("a2.acoustic.ch1@1") is extract_ch1
    assert list(extract(_example())[0]) == list(FEATURE_ORDER)
    assert all(name.startswith("a2.acoustic.ch0.") for name in FEATURE_ORDER)


def test_features_espectrales_son_finitas_y_bandas_suman_uno():
    features, diagnostics = extract(_example())

    assert set(features) == set(FEATURE_ORDER)
    assert all(np.isfinite(value) for value in features.values())
    bands = [value for name, value in features.items() if ".band_" in name]
    assert np.isclose(sum(bands), 1.0, atol=1e-5)
    assert diagnostics["a2.acoustic.ch0.raw_rms"] > 0
    assert np.isfinite(diagnostics["a2.acoustic.ch0.raw_rms"])


def test_seleccion_ch1_es_independiente_y_no_contrastivo():
    features, diagnostics = extract_ch1(_example(ch1=True))

    assert all(name.startswith("a2.acoustic.ch1.") for name in features)
    assert all(name.startswith("a2.acoustic.ch1.") for name in diagnostics)
    assert not any("contrast" in name or "delta" in name for name in features)


def test_audio_silencioso_y_corto_tiene_semantica_explicita():
    for ex in (AudioExample(np.zeros(10, dtype=np.float32), None),
               AudioExample(np.zeros(2048, dtype=np.float32), None)):
        features, diagnostics = extract(ex)
        assert all(np.isfinite(value) for value in features.values())
        assert diagnostics["a2.acoustic.ch0.is_silent"] == 1.0
        assert diagnostics["a2.acoustic.ch0.n_frames"] >= 1.0
        assert diagnostics["a2.acoustic.ch0.n_active_frames"] == 0.0


def test_frames_incluyen_la_cola_parcial_de_audio_largo():
    signal = np.zeros(257, dtype=np.float32)
    signal[-1] = 1.0

    frames, short = _frames(signal)

    assert short is False
    assert frames.shape == (2, 256)
    assert frames[1, 128] == 1.0


def test_audio_no_finito_no_propaga_features_no_finitas():
    signal = np.zeros(8000, dtype=np.float32)
    signal[0] = np.nan

    features, diagnostics = extract(AudioExample(signal, None))

    assert all(np.isfinite(value) for value in features.values())
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_speech_and_silence_variants_use_the_injected_energy_mask():
    signal = np.concatenate((np.full(400, 0.2), np.full(400, 0.01))).astype(np.float32)
    segmentation = Segmentation((Turn(0, 0.0, 0.05),), "energy@1", {})
    ex = AudioExample(signal, None, seg=segmentation)

    speech, speech_diag = extract_speech(ex)
    silence, silence_diag = extract_silence(ex)

    assert set(speech) == set(SPEECH_FEATURE_ORDER)
    assert set(silence) == set(SILENCE_FEATURE_ORDER)
    assert speech_diag["a2.acoustic.ch0.speech.mask_source"] == "energy@1"
    assert silence_diag["a2.acoustic.ch0.silence.mask_source"] == "energy@1"
    assert speech_diag["a2.acoustic.ch0.speech.n_region_frames"] > 0
    assert silence_diag["a2.acoustic.ch0.silence.n_region_frames"] > 0


def test_region_variants_reject_missing_segmentation_and_declare_empty_regions():
    with np.testing.assert_raises_regex(ValueError, "segmentacion"):
        extract_speech(_example())

    ex = AudioExample(
        np.full(800, 0.2, dtype=np.float32),
        None,
        seg=Segmentation((), "energy@1", {}),
    )
    features, diagnostics = extract_speech(ex)

    assert all(np.isfinite(value) for value in features.values())
    assert diagnostics["a2.acoustic.ch0.speech.empty_region"] == 1.0
    assert diagnostics["a2.acoustic.ch0.speech.n_region_frames"] == 0.0


def test_region_variants_have_fixed_non_colliding_namespaces_and_are_registered():
    from altur.registry import extractors

    assert extractors.resolve("a2.acoustic.ch0.speech@1") is extract_speech
    assert extractors.resolve("a2.acoustic.ch0.silence@1") is extract_silence
    assert set(SPEECH_FEATURE_ORDER).isdisjoint(SILENCE_FEATURE_ORDER)
    assert all(name.startswith("a2.acoustic.ch0.speech.") for name in SPEECH_FEATURE_ORDER)
    assert all(name.startswith("a2.acoustic.ch0.silence.") for name in SILENCE_FEATURE_ORDER)
