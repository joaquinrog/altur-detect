"""Guarda: el código incorporado de otros repos no trae identificadores del dataset (D-A5.1).

Un `anon_id` junto con su etiqueta es una fila de la hoja de respuestas, y los términos de Altur
prohíben redistribuir. Este repo se vuelve público y el historial no se despublica (D-A5.2).
"""

from __future__ import annotations

from pathlib import Path

from altur import bundle as bundle_mod

RESEARCH = Path(__file__).resolve().parent.parent / "research"
_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def _files() -> list[Path]:
    return [p for p in RESEARCH.rglob("*") if p.is_file() and not p.is_symlink()]


def test_research_existe():
    assert RESEARCH.is_dir() and _files(), "research/ desapareció o quedó vacío"


def test_research_no_contiene_anon_id():
    hits = [
        str(p.relative_to(RESEARCH))
        for p in _files()
        if bundle_mod._ANON_ID.search(p.read_bytes().decode("utf-8", errors="ignore"))
    ]
    assert not hits, f"identificadores de llamada bajo research/: {hits}"


def test_research_no_contiene_audio():
    audio = [str(p.relative_to(RESEARCH)) for p in _files() if p.suffix.lower() in _AUDIO]
    assert not audio, f"audio bajo research/: {audio}"
