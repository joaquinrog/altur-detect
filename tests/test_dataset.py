"""Tests de `dataset.py`. El foco es el candado anti-fuga, no el happy path.

Corren sin el dataset real: construyen un `data/` de juguete. Los que exigen los 353 WAV
se marcan y se saltan si no estan descargados, para que CI no dependa de 671 MB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from altur.dataset import LABELS, Dataset, DatasetError, read_manifest, read_turns
from altur.io import write_wav
from altur.types import AudioExample, DatasetRecord

ROWS = [
    ("call_a", "human", "train", 61.0),
    ("call_b", "synthetic", "train", 70.5),
    ("call_c", "synthetic", "val", 80.0),
]


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "audio").mkdir(parents=True)
    (root / "turns").mkdir(parents=True)
    lines = ["anon_id,label,split,duration_s"]
    rng = np.random.default_rng(0)
    for cid, label, split, dur in ROWS:
        lines.append(f"{cid},{label},{split},{dur}")
        n = int(8000 * 2.0)
        write_wav(str(root / "audio" / f"{cid}.wav"),
                  rng.normal(0, 0.05, n).astype(np.float32),
                  rng.normal(0, 0.05, n).astype(np.float32))
        (root / "turns" / f"{cid}.json").write_text(
            json.dumps({"turns": [{"channel": 1, "start": 0.5, "end": 1.0},
                                  {"channel": 0, "start": 0.0, "end": 0.4}]})
        )
    (root / "manifest.csv").write_text("\n".join(lines) + "\n")
    return root


def test_carga_e_indice(toy: Path) -> None:
    ds = Dataset(toy)
    assert len(ds) == 3
    assert ds.ids == ("call_a", "call_b", "call_c")       # orden determinista
    assert ds.ids_for("train") == ("call_a", "call_b")
    assert ds.ids_for("val") == ("call_c",)


def test_synthetic_es_la_clase_positiva(toy: Path) -> None:
    """Si esto se invierte, toda metrica mide lo contrario de lo que pregunta el contrato."""
    assert LABELS["synthetic"] == 1 and LABELS["human"] == 0
    ds = Dataset(toy)
    assert ds.record("call_b").label == 1
    assert ds.record("call_a").label == 0


def test_load_no_devuelve_metadatos(toy: Path) -> None:
    """🔴 El candado: lo que ve un extractor no lleva id, etiqueta, split ni procedencia."""
    ex = Dataset(toy).load("call_b")
    assert isinstance(ex, AudioExample)
    campos = set(ex.__slots__)
    assert not campos & {"example_id", "label", "split", "groups", "provenance"}
    for atributo in ("example_id", "label", "split", "groups", "provenance"):
        assert not hasattr(ex, atributo)


def test_record_y_load_son_llamadas_distintas(toy: Path) -> None:
    ds = Dataset(toy)
    assert isinstance(ds.record("call_a"), DatasetRecord)
    assert isinstance(ds.load("call_a"), AudioExample)


def test_audio_es_read_only(toy: Path) -> None:
    ex = Dataset(toy).load("call_a")
    assert not ex.ch0.flags.writeable
    with pytest.raises(ValueError):
        ex.ch0[0] = 1.0


def test_turns_solo_si_se_piden_y_van_marcados(toy: Path) -> None:
    """El camino comodo debe parecerse a produccion: en `/detect` no hay `turns/`."""
    ds = Dataset(toy)
    assert ds.load("call_a").seg is None
    seg = ds.load("call_a", with_turns=True).seg
    assert seg is not None and seg.is_oracle
    assert [t.start for t in seg.turns] == [0.0, 0.5]     # ordenado, no como venia


def test_fingerprint_cambia_si_cambia_una_etiqueta(toy: Path, tmp_path: Path) -> None:
    fp = Dataset(toy).fingerprint()
    m = toy / "manifest.csv"
    m.write_text(m.read_text().replace("call_a,human", "call_a,synthetic"))
    assert Dataset(toy).fingerprint() != fp


def test_manifest_invalido_falla_fuerte(toy: Path) -> None:
    m = toy / "manifest.csv"
    original = m.read_text()
    for roto, _motivo in [
        (original.replace("call_b,synthetic", "call_a,synthetic"), "id duplicado"),
        (original.replace(",human,", ",humano,"), "label desconocida"),
        (original.replace(",train,", ",entrena,"), "split desconocido"),
        (original.replace(",61.0", ",sesenta"), "duracion no numerica"),
        (original.replace("anon_id,", "id,"), "falta columna"),
    ]:
        m.write_text(roto)
        with pytest.raises(DatasetError):
            read_manifest(m)
    m.write_text(original)


def test_id_inexistente(toy: Path) -> None:
    with pytest.raises(DatasetError, match="no existe en el manifest"):
        Dataset(toy).record("call_zzz")


def test_falta_audio(toy: Path) -> None:
    (toy / "audio" / "call_b.wav").unlink()
    with pytest.raises(DatasetError, match="faltan"):
        Dataset(toy)
    Dataset(toy, require_audio=False)          # los metadatos siguen sirviendo


def test_turns_como_lista_suelta(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"channel": 0, "start": 1.0, "end": 2.0}]))
    assert len(read_turns(p).turns) == 1
