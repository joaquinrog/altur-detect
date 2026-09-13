"""Descarga y verifica el dataset oficial de Altur.

Dos piezas, dos fuentes:

  - `manifest.csv` + `turns/` viven en el REPO (github.com/alturio/hackmty26, commit 429adf7).
  - `audio/*.wav` vive en el RELEASE v1.0, como `altur-challenge-audio.zip` (671 MB).

El SHA-256 del zip esta congelado abajo. **Verificarlo no es ceremonia**: el asset se sirve
desde un CDN, un corte a medias produce un zip que descomprime "bien" hasta que no, y una
corrida de tres personas sobre bytes distintos no es comparable. Si el hash no cuadra, el
script borra lo descargado y falla: no deja a medias algo que parezca completo.

FACT (notas/01 §1): tamano 671 269 556 B, sha256 41e597bd...e936bb, un solo tag.

Uso:
    python scripts/download_data.py            # descarga lo que falte y verifica
    python scripts/download_data.py --check    # solo verifica lo ya descargado
    python scripts/download_data.py --force    # re-descarga aunque exista
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "alturio/hackmty26"
COMMIT = "429adf76b15d1bd18e26b50f371ca4f13b0585c0"
RELEASE_TAG = "v1.0"
ASSET = "altur-challenge-audio.zip"

AUDIO_SHA256 = "41e597bd87dbe76b23b6fffb958f3fbc3761e3fe355a123dff53625f08e936bb"
AUDIO_BYTES = 671_269_556

# FACT (notas/05 §3.3): el manifest trae 353 filas, no las "~300" de la diapositiva.
EXPECTED_CALLS = 353

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio"
TURNS_DIR = DATA / "turns"
MANIFEST = DATA / "manifest.csv"

CHUNK = 1 << 20


class DataError(RuntimeError):
    pass


def sha256_file(path: Path, *, progress: bool = False) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                _bar("verificando", done, total)
    if progress:
        print()
    return h.hexdigest()


def _bar(label: str, done: int, total: int) -> None:
    pct = 100 * done / total
    mb = done / 1e6
    tot_mb = total / 1e6
    print(f"\r  {label}: {pct:5.1f}%  {mb:7.1f}/{tot_mb:.1f} MB", end="", flush=True)


def download(url: str, dest: Path, *, expected_bytes: int | None = None) -> None:
    """Descarga a un `.part` y renombra al final.

    El `.part` importa: si el proceso muere a medias, el archivo final nunca llega a
    existir, asi que una corrida posterior no lo confunde con una descarga completa.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {url}")
    try:
        with urllib.request.urlopen(url) as resp:
            total = expected_bytes or int(resp.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as f:
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        _bar("descargando", done, total)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise DataError(f"fallo la descarga de {url}: {e}") from e
    print()
    if expected_bytes is not None and tmp.stat().st_size != expected_bytes:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise DataError(f"tamano inesperado: {got} B, esperaba {expected_bytes} B")
    tmp.replace(dest)


def fetch_repo_metadata(*, force: bool) -> None:
    """`manifest.csv` y `turns/` desde el tarball del repo, fijado al commit."""
    if MANIFEST.exists() and TURNS_DIR.exists() and not force:
        print("- metadata del repo: ya esta")
        return
    print("- metadata del repo (manifest.csv + turns/)")
    tgz = DATA / f"repo-{COMMIT}.tar.gz"
    download(f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}", tgz)

    with tarfile.open(tgz) as tf:
        members = tf.getmembers()
        root = members[0].name.split("/")[0]
        TURNS_DIR.mkdir(parents=True, exist_ok=True)
        found_manifest = False
        for m in members:
            rel = m.name[len(root) + 1 :]
            if not m.isfile():
                continue
            if rel == "manifest.csv":
                _extract_to(tf, m, MANIFEST)
                found_manifest = True
            elif rel.startswith("turns/") and rel.endswith(".json"):
                _extract_to(tf, m, TURNS_DIR / Path(rel).name)
    tgz.unlink(missing_ok=True)
    if not found_manifest:
        raise DataError(f"el tarball del repo no traia manifest.csv (commit {COMMIT})")


def _extract_to(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    src = tf.extractfile(member)
    if src is None:
        raise DataError(f"no se pudo leer {member.name} del tarball")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def fetch_audio(*, force: bool) -> None:
    zip_path = DATA / ASSET
    n_wav = len(list(AUDIO_DIR.glob("*.wav"))) if AUDIO_DIR.exists() else 0
    if n_wav >= EXPECTED_CALLS and not force:
        print(f"- audio: ya estan los {n_wav} WAV")
        return

    if not zip_path.exists() or force:
        print(f"- audio ({AUDIO_BYTES / 1e6:.0f} MB)")
        url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET}"
        download(url, zip_path, expected_bytes=AUDIO_BYTES)

    print("- verificando SHA-256 del zip")
    got = sha256_file(zip_path, progress=True)
    if got != AUDIO_SHA256:
        zip_path.unlink(missing_ok=True)
        raise DataError(
            "SHA-256 no coincide. El zip quedo BORRADO a proposito: un archivo corrupto "
            "que sobrevive es peor que no tenerlo.\n"
            f"  esperaba {AUDIO_SHA256}\n  obtuve   {got}"
        )
    print(f"  ok  {got}")

    print("- descomprimiendo")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.endswith(".wav") or name.startswith("."):
                continue
            # Nombre plano a proposito: un zip no confiable no dicta rutas (zip-slip).
            with zf.open(info) as src, (AUDIO_DIR / name).open("wb") as out:
                shutil.copyfileobj(src, out)
    print(f"  {len(list(AUDIO_DIR.glob('*.wav')))} WAV en {AUDIO_DIR.relative_to(ROOT)}")


CHECKER = DATA / "official" / "check_endpoint.py"
# FACT: SHA-256 de `scripts/check_endpoint.py` en alturio/hackmty26@429adf7. Es el cliente con
# el que juzgan; `scripts/e2e_judge.py` se niega a correr con cualquier otro.
CHECKER_SHA256 = "593f78ceb80017e791f0f8d552ca6a7b3b6763c11beedfa1f68ab1a55c363364"


def fetch_official_checker(*, force: bool) -> None:
    """El cliente oficial del juez, fijado al commit y verificado. Vive en `data/`: no se versiona."""
    if CHECKER.exists() and not force and sha256_file(CHECKER) == CHECKER_SHA256:
        print("- cliente oficial del juez: ya esta")
        return
    print("- cliente oficial del juez (scripts/check_endpoint.py)")
    download(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/scripts/check_endpoint.py", CHECKER)
    got = sha256_file(CHECKER)
    if got != CHECKER_SHA256:
        CHECKER.unlink(missing_ok=True)
        raise DataError(
            f"SHA-256 del cliente oficial no coincide\n  esperaba {CHECKER_SHA256}\n  obtuve   {got}"
        )


def verify() -> int:
    """Comprueba que las tres piezas cuadran entre si. Devuelve el codigo de salida."""
    problems: list[str] = []

    if not MANIFEST.exists():
        problems.append("falta manifest.csv")
        rows = []
    else:
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != EXPECTED_CALLS:
            problems.append(f"manifest.csv trae {len(rows)} filas, esperaba {EXPECTED_CALLS}")

    ids = {r["anon_id"] for r in rows} if rows else set()
    wavs = {p.stem for p in AUDIO_DIR.glob("*.wav")} if AUDIO_DIR.exists() else set()
    turns = {p.stem for p in TURNS_DIR.glob("*.json")} if TURNS_DIR.exists() else set()

    if ids:
        if missing := sorted(ids - wavs)[:5]:
            problems.append(f"WAV faltantes ({len(ids - wavs)}), p.ej. {missing}")
        if extra := sorted(wavs - ids)[:5]:
            problems.append(f"WAV que no estan en el manifest ({len(wavs - ids)}): {extra}")
        if missing_t := sorted(ids - turns)[:5]:
            problems.append(f"turns/ faltantes ({len(ids - turns)}), p.ej. {missing_t}")

    print("\nResumen")
    print(f"  manifest.csv  {len(rows)} filas")
    print(f"  audio/        {len(wavs)} WAV")
    print(f"  turns/        {len(turns)} JSON")
    if rows:
        by_label: dict[str, int] = {}
        by_split: dict[str, int] = {}
        for r in rows:
            by_label[r["label"]] = by_label.get(r["label"], 0) + 1
            by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        print(f"  label         {dict(sorted(by_label.items()))}")
        print(f"  split         {dict(sorted(by_split.items()))}")

    if problems:
        print("\nPROBLEMAS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nTodo cuadra.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="solo verificar, no descargar")
    ap.add_argument("--force", action="store_true", help="re-descargar aunque ya exista")
    ap.add_argument("--keep-zip", action="store_true", help="no borrar el zip tras extraer")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if not args.check:
        try:
            fetch_repo_metadata(force=args.force)
            fetch_official_checker(force=args.force)
            fetch_audio(force=args.force)
        except DataError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 2
        if not args.keep_zip:
            (DATA / ASSET).unlink(missing_ok=True)

    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
