"""D-A5.7: Altur prohíbe usar el detector de watermark del proveedor TTS ("es como trampa")."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PROHIBIDOS = ("watermark",)


def test_ni_el_codigo_ni_las_dependencias_usan_el_detector_de_watermark():
    archivos = [*(RAIZ / "src").rglob("*.py"), RAIZ / "pyproject.toml"]
    hallazgos = [
        f"{archivo.relative_to(RAIZ)}: {termino}"
        for archivo in archivos
        if archivo.exists()
        for termino in PROHIBIDOS
        if termino in archivo.read_text(encoding="utf-8").lower()
    ]
    assert not hallazgos, hallazgos
