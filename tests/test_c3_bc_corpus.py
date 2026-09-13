from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "build_c3_bc_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_c3_bc_corpus", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _wav(root: Path, call_id: str) -> None:
    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    (audio / f"{call_id}.wav").write_bytes(f"wav:{call_id}".encode())


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    official, block_b, block_b2 = (tmp_path / name for name in ("official", "b", "b2"))
    _write_csv(
        official / "manifest.csv",
        ["anon_id", "label", "split", "duration_s"],
        [
            {"anon_id": "off-h", "label": "human", "split": "train", "duration_s": 2},
            {"anon_id": "off-s", "label": "synthetic", "split": "train", "duration_s": 2},
            {"anon_id": "off-v", "label": "human", "split": "val", "duration_s": 2},
        ],
    )
    for call_id in ("off-h", "off-s", "off-v"):
        _wav(official, call_id)

    _write_csv(
        block_b / "manifest.csv",
        ["anon_id", "label", "split", "duration_s"],
        [
            {"anon_id": call_id, "label": label, "split": "hidden", "duration_s": 2}
            for call_id, label in (
                ("b-sp", "synthetic"), ("b-hp", "human"),
                ("b-sg", "synthetic"), ("b-hg", "human"),
            )
        ],
    )
    _write_csv(
        block_b / "pairs.csv",
        ["guion", "variante", "id_sintetica", "id_humana", "modelo_tts", "voz", "persona"],
        [
            {"guion": "G1", "variante": "plana", "id_sintetica": "b-sp", "id_humana": "b-hp",
             "modelo_tts": "m", "voz": "voice-shared", "persona": "person-1"},
            {"guion": "G1", "variante": "g711", "id_sintetica": "b-sg", "id_humana": "b-hg",
             "modelo_tts": "m", "voz": "voice-shared", "persona": "person-1"},
        ],
    )
    for call_id in ("b-sp", "b-hp", "b-sg", "b-hg"):
        _wav(block_b, call_id)

    _write_csv(
        block_b2 / "manifest.csv",
        ["anon_id", "label", "split", "duration_s"],
        [
            {"anon_id": "b2-p", "label": "synthetic", "split": "hidden", "duration_s": 2},
            {"anon_id": "b2-g", "label": "synthetic", "split": "hidden", "duration_s": 2},
        ],
    )
    _write_csv(
        block_b2 / "voices_map.csv",
        ["anon_id", "slot", "guion", "variante", "modelo", "genero", "voz"],
        [
            {"anon_id": "b2-p", "slot": "slot-1", "guion": "G1", "variante": "plana",
             "modelo": "m", "genero": "f", "voz": "voice-shared"},
            {"anon_id": "b2-g", "slot": "slot-1", "guion": "G1", "variante": "g711",
             "modelo": "m", "genero": "f", "voz": "voice-shared"},
        ],
    )
    for call_id in ("b2-p", "b2-g"):
        _wav(block_b2, call_id)
    return official, block_b, block_b2


def test_index_uses_train_and_flat_only_and_connects_private_relations(tmp_path: Path):
    corpus = _load_module()
    official, block_b, block_b2 = _fixture(tmp_path)

    index = corpus.build_corpus_index(
        official_root=official,
        block_b_root=block_b,
        block_b2_root=block_b2,
        expected_counts={"total": 5, "human": 2, "synthetic": 3},
    )

    assert {row.source for row in index.rows} == {"official_train", "bloque_b", "bloque_b2"}
    assert not any(row.call_id in {"off-v", "b-sg", "b-hg", "b2-g"} for row in index.rows)
    private = {row.call_id: row for row in index.rows if row.source != "official_train"}
    assert {private[key].group_id for key in ("b-sp", "b-hp", "b2-p")} == {
        private["b-sp"].group_id
    }
    assert len(index.fingerprint) == 64
    assert all(len(row.audio_sha256) == 64 for row in index.rows)


def test_index_fails_closed_when_a_joined_audio_is_missing(tmp_path: Path):
    corpus = _load_module()
    official, block_b, block_b2 = _fixture(tmp_path)
    (block_b2 / "audio" / "b2-p.wav").unlink()

    with pytest.raises(corpus.C3CorpusError, match="WAV|audio"):
        corpus.build_corpus_index(
            official_root=official,
            block_b_root=block_b,
            block_b2_root=block_b2,
            expected_counts={"total": 5, "human": 2, "synthetic": 3},
        )


def test_index_fails_closed_on_non_unique_private_join(tmp_path: Path):
    corpus = _load_module()
    official, block_b, block_b2 = _fixture(tmp_path)
    with (block_b2 / "voices_map.csv").open("a", encoding="utf-8") as stream:
        stream.write("b2-p,slot-2,G2,plana,m,m,other-voice\n")

    with pytest.raises(corpus.C3CorpusError, match="duplicad|1:1"):
        corpus.build_corpus_index(
            official_root=official,
            block_b_root=block_b,
            block_b2_root=block_b2,
            expected_counts={"total": 5, "human": 2, "synthetic": 3},
        )
