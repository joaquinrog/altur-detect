"""
Splits GroupKFold — ver CONTRATOS.md seccion 1.

Este archivo es el "fallback local": genera data/splits.csv con el mismo formato
que va a entregar Persona 1. El dia que Persona 1 comparta el suyo, se reemplaza
ese CSV y nada mas en el proyecto cambia (todos los demas scripts solo leen
data/splits.csv, nunca llaman a GroupKFold directamente).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "manifest.csv"
DATA_DIR = Path(__file__).parent / "data"
SPLITS_PATH = DATA_DIR / "splits.csv"

N_FOLDS = 5
SEED = 42  # documentado: cualquiera que regenere este archivo con esta seed
           # obtiene los mismos folds, asi que aunque no sea el CSV oficial de
           # Persona 1, es reproducible entre maquinas mientras tanto.


def generate_splits(force=False):
    """Genera data/splits.csv si no existe (o si force=True). Devuelve el DataFrame."""
    DATA_DIR.mkdir(exist_ok=True)
    if SPLITS_PATH.exists() and not force:
        return pd.read_csv(SPLITS_PATH)

    manifest = pd.read_csv(MANIFEST_PATH)
    # GroupKFold sobre las 353 llamadas completas (train+val originales del
    # manifest de Altur), agrupando por anon_id. El split train/val de Altur no
    # se preserva aqui a proposito: es el propio protocolo de CV del equipo,
    # el set oculto de evaluacion del reto es otra cosa que no controlamos.
    gkf = GroupKFold(n_splits=N_FOLDS)
    manifest = manifest.sort_values("anon_id").reset_index(drop=True)
    fold_col = pd.Series(-1, index=manifest.index, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(manifest, groups=manifest["anon_id"])
    ):
        fold_col.iloc[val_idx] = fold_idx

    out = manifest[["anon_id", "label"]].copy()
    out["fold"] = fold_col
    out.to_csv(SPLITS_PATH, index=False)
    return out


def load_splits():
    """Punto de entrada que deben usar todos los demas scripts."""
    if not SPLITS_PATH.exists():
        return generate_splits()
    return pd.read_csv(SPLITS_PATH)


if __name__ == "__main__":
    df = generate_splits(force=True)
    print(f"Splits generados en {SPLITS_PATH}")
    print(df.groupby(["fold", "label"]).size().unstack(fill_value=0))
    # verificacion: ningun anon_id debe repetirse entre folds (GroupKFold ya lo
    # garantiza, esto solo lo confirma explicitamente)
    dup = df.groupby("anon_id")["fold"].nunique()
    assert (dup == 1).all(), "anon_id repetido en mas de un fold — no deberia pasar con GroupKFold"
    print("OK: cada anon_id aparece en exactamente un fold.")
