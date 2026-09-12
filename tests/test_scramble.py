"""`behavior_scramble@1` (A3.5) y el candado de segmentación entre condiciones."""

import numpy as np
import pytest

import altur.seg.scramble  # noqa: F401  — registra el turn_source
from altur.registry import turn_sources
from altur.seg.scramble import behavior_scramble
from altur.seg.vad import energy
from altur.types import AudioExample, Segmentation, Turn

# A 8 kHz el VAD de energía exige turnos de >= 200 ms y funde huecos de < 450 ms. Un fixture
# con ráfagas de milisegundos no produce ni un turno: los tramos van a escala de llamada.
_TURNOS_CH0 = [(8_000, 24_000), (40_000, 56_000), (96_000, 112_000)]
_TURNOS_CH1 = [(28_000, 38_000), (64_000, 80_000), (120_000, 136_000)]
_N = 145_000  # ~18 s


def _conversacion(seed: int = 0, n: int = _N) -> AudioExample:
    """Dos canales alternando habla, con silencios de duración DESIGUAL.

    Desiguales a propósito: permutar huecos idénticos no cambiaría nada y el test pasaría
    sin probar nada.
    """
    rng = np.random.default_rng(seed)
    ch0 = (rng.standard_normal(n) * 1e-4).astype(np.float32)
    ch1 = (rng.standard_normal(n) * 1e-4).astype(np.float32)
    for destino, tramos in ((ch0, _TURNOS_CH0), (ch1, _TURNOS_CH1)):
        for ini, fin in tramos:
            destino[ini:fin] = (rng.standard_normal(fin - ini) * 0.3).astype(np.float32)
    return AudioExample(ch0=ch0, ch1=ch1)


def _clave(seg: Segmentation):
    return [(t.channel, round(t.start, 9), round(t.end, 9)) for t in seg.turns]


def test_el_audio_no_se_toca():
    """Es la condición que define la prueba: si cambia el audio, no mide comportamiento."""
    ex = _conversacion()
    antes0, antes1 = np.array(ex.ch0, copy=True), np.array(ex.ch1, copy=True)
    behavior_scramble(ex)
    assert np.array_equal(np.asarray(ex.ch0), antes0)
    assert np.array_equal(np.asarray(ex.ch1), antes1)


def test_conserva_las_duraciones_y_el_numero_de_turnos():
    ex = _conversacion()
    base, rev = energy(ex), behavior_scramble(ex)
    assert len(rev.turns) == len(base.turns)
    for canal in (0, 1):
        d_base = sorted(round(t.end - t.start, 9) for t in base.turns if t.channel == canal)
        d_rev = sorted(round(t.end - t.start, 9) for t in rev.turns if t.channel == canal)
        assert d_base == d_rev


def test_mueve_los_tiempos():
    ex = _conversacion(seed=3)
    base, rev = energy(ex), behavior_scramble(ex)
    assert _clave(base) != _clave(rev), "si los turnos salen idénticos, no revolvió nada"


def test_es_determinista_por_audio():
    ex = _conversacion()
    assert _clave(behavior_scramble(ex)) == _clave(behavior_scramble(ex))


def test_audios_distintos_dan_revueltos_distintos():
    a, b = _conversacion(seed=1), _conversacion(seed=2)
    assert _clave(behavior_scramble(a)) != _clave(behavior_scramble(b))


def test_los_turnos_caben_en_la_llamada():
    ex = _conversacion()
    for t in behavior_scramble(ex).turns:
        assert 0.0 <= t.start <= t.end <= ex.duration_s + 1e-9


def test_se_declara_en_la_segmentacion_y_en_el_registro():
    ex = _conversacion()
    seg = behavior_scramble(ex)
    assert seg.source == "behavior_scramble@1"
    assert not seg.is_oracle
    assert seg.params["preserves_audio"] is True
    assert seg.params["base_source"] == "energy@1"

    meta = turn_sources.meta("behavior_scramble@1")
    assert meta["is_oracle"] is False
    assert meta["mode"] == "permute_gaps"


def test_el_fixture_produce_turnos_en_ambos_canales():
    """Guarda: si el VAD no detecta nada, los demás tests pasarían sin probar nada."""
    base = energy(_conversacion())
    for canal in (0, 1):
        assert len([t for t in base.turns if t.channel == canal]) >= 2


def test_un_canal_con_menos_de_dos_turnos_se_deja_igual():
    """Sin al menos dos turnos no hay huecos que permutar; inventar uno sería fabricar señal."""
    rng = np.random.default_rng(0)
    n = 40_000
    ch0 = (rng.standard_normal(n) * 1e-4).astype(np.float32)
    ch0[8_000:24_000] = (rng.standard_normal(16_000) * 0.3).astype(np.float32)
    ex = AudioExample(ch0=ch0, ch1=np.zeros(n, dtype=np.float32))
    base, rev = energy(ex), behavior_scramble(ex)
    assert len([t for t in base.turns if t.channel == 0]) == 1
    assert _clave(base) == _clave(rev)


def test_cambia_las_features_conductuales():
    from altur.features.behavioral import extract

    ex = _conversacion(seed=5)
    limpio = AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=energy(ex))
    revuelto = AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=behavior_scramble(ex))
    f_a, _ = extract(limpio)
    f_b, _ = extract(revuelto)
    assert f_a != f_b, "revolver los turnos tiene que mover el eje conductual"


def test_no_cambia_las_features_acusticas_que_no_usan_segmentacion():
    """El audio es el mismo, así que un extractor con needs_seg=False no puede notar nada."""
    from altur.features.acoustic_minimal import extract

    ex = _conversacion(seed=6)
    a, _ = extract(AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=energy(ex)))
    b, _ = extract(AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=behavior_scramble(ex)))
    assert a == b


def test_el_runner_nunca_arrastra_la_segmentacion_de_entrada():
    """Requisito de A3.5: un transform que mueve tiempos obliga a recalcular la segmentación.

    El arnés lo garantiza por construcción — `runner.py` corre el segmentador sobre el audio
    ya transformado y descarta el `seg` que traiga el ejemplo. Este test fija esa propiedad
    para que nadie la "optimice" reusando la segmentación de entrada.
    """
    ex = _conversacion(seed=7)
    mentira = Segmentation((Turn(0, 0.0, 0.01),), source="mentira@1")
    con_mentira = AudioExample(ex.ch0, ex.ch1, sr=ex.sr, seg=mentira)
    recalculado = energy(con_mentira)
    assert recalculado.source == "energy@1"
    assert _clave(recalculado) == _clave(energy(ex))
    assert _clave(recalculado) != _clave(mentira)


def test_no_es_un_transform_sino_un_turn_source():
    """Un transform no puede alterar turnos: el runner descarta su seg (runner.py:243)."""
    from altur.registry import RegistryError, transforms

    assert "behavior_scramble@1" in turn_sources
    with pytest.raises(RegistryError, match="no registrado"):
        transforms.get("behavior_scramble@1")
