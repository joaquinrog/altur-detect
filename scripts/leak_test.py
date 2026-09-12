"""Test de fuga: permutar etiquetas y verificar que el AUC cae a ~0.5.

El contrato de `nested_cv` dice que ninguna decision aprendida toca el fold externo antes
de predecirlo. Esto lo COMPRUEBA en vez de afirmarlo: si al destruir la relacion entre X e
y el AUC sigue alto, la unica senal que queda es la que el protocolo no debio dejar pasar.

Usa features baratas y reales (numpy puro sobre ch0), no ruido sintetico, porque un test
sobre datos de juguete no ejercita el camino de carga, ni la cache, ni los folds reales.

    python scripts/leak_test.py            # real + permutado
    python scripts/leak_test.py --quick    # 80 llamadas, para iterar

INTERPRETACION:
  · AUC real alto y AUC permutado ~0.5  -> el protocolo aguanta.
  · AUC permutado claramente > 0.5      -> hay fuga. NO seguir hasta entenderlo.
  · AUC real ~0.5 tambien               -> features inutiles; el test no dice nada del protocolo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from altur.dataset import Dataset
from altur.protocol import (
    grouped_inner_splits,
    load_groups,
    nested_cv,
    permute_labels_within_groups,
)
from altur.types import AudioExample

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "leaktest_features_v1.npz"

FRAME, HOP = 256, 128  # 32 ms / 16 ms a 8 kHz
FEATURE_NAMES = (
    "log_rms", "zcr", "centroid_hz", "rolloff85_hz",
    "log_flatness", "frac_sub300", "frac_over3400", "log_rms_iqr",
)


def features(ex: AudioExample) -> np.ndarray:
    """8 descriptores espectrales baratos sobre ch0. Solo numpy, a proposito.

    Incluye `frac_sub300` porque el audit forense midio que el sintetico tiene el triple
    de energia por DEBAJO de la banda telefonica (notas/11 §8). Si algo separa barato,
    probablemente es eso — y probablemente sea canal, no voz.
    """
    x = np.asarray(ex.ch0, dtype=np.float64)
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(FRAME)[None, :]

    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    power = spec**2
    freqs = np.fft.rfftfreq(FRAME, 1 / ex.sr)
    total = power.sum(axis=1)

    rms = np.sqrt((frames**2).mean(axis=1)) + 1e-12
    voiced = rms > np.percentile(rms, 40)      # descarta silencio sin VAD: esto no es el detector
    if voiced.sum() < 10:
        voiced = np.ones(n, dtype=bool)

    centroid = (power * freqs[None, :]).sum(axis=1) / total
    cum = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    flatness = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    zcr = (np.diff(np.sign(frames), axis=1) != 0).mean(axis=1)
    sub300 = power[:, freqs < 300].sum(axis=1) / total
    over3400 = power[:, freqs > 3400].sum(axis=1) / total

    lr = np.log(rms)
    return np.array([
        lr[voiced].mean(),
        zcr[voiced].mean(),
        centroid[voiced].mean(),
        float(rolloff[voiced].mean()),
        np.log(flatness[voiced] + 1e-12).mean(),
        sub300[voiced].mean(),
        over3400[voiced].mean(),
        float(np.subtract(*np.percentile(lr, [75, 25]))),
    ], dtype=np.float64)


def build_matrix(ds: Dataset, ids: tuple[str, ...], *, use_cache: bool = True) -> np.ndarray:
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == list(ids):
            print(f"  features desde cache ({CACHE.relative_to(ROOT)})")
            return z["X"]
    X = np.zeros((len(ids), len(FEATURE_NAMES)))
    t0 = time.time()
    for i, cid in enumerate(ids):
        X[i] = features(ds.load(cid))
        if (i + 1) % 25 == 0 or i + 1 == len(ids):
            el = time.time() - t0
            print(f"\r  extrayendo {i + 1}/{len(ids)}  {el:5.1f}s", end="", flush=True)
    print()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, ids=np.array(ids))
    return X


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC por rangos. Sin sklearn.metrics a proposito: una dependencia menos que auditar."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # empates comparten el rango medio
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def make_fit_predict(ds: Dataset, X: np.ndarray, ids: tuple[str, ...], y: np.ndarray):
    """Ajusta TODO dentro de `IN_k` y predice `OUT_k` una sola vez."""
    pos = {cid: i for i, cid in enumerate(ids)}

    def fit_predict(train_ids, test_ids, g_in):
        tr = np.array([pos[i] for i in train_ids])
        te = np.array([pos[i] for i in test_ids])
        y_tr = y[tr]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        model.fit(X[tr], y_tr)

        # Umbral: se fija con folds INTERNOS por grupo, nunca con el fold externo.
        inner = grouped_inner_splits(y_tr, g_in, n_folds=3)
        oof = np.full(len(tr), np.nan)
        for a, b in inner:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr][a], y_tr[a])
            oof[b] = m.predict_proba(X[tr][b])[:, 1]
        thr = float(np.nanmedian(oof)) if np.isfinite(oof).any() else 0.5

        return model.predict_proba(X[te])[:, 1], thr, {"n_train": len(tr)}

    return fit_predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="solo 80 llamadas")
    ap.add_argument("--tol", type=float, default=0.60,
                    help="AUC permutado maximo tolerado (default 0.60)")
    args = ap.parse_args()

    ds = Dataset()
    groups = load_groups()
    from altur.protocol import get_protocol

    proto = get_protocol("official_v1", ds)
    ids = proto.fit_ids[:80] if args.quick else proto.fit_ids
    if args.quick:
        proto = type(proto)(name=proto.name, fit_ids=ids, judge_ids=(), n_folds=3)

    print(f"Test de fuga · {proto.name} · {len(ids)} llamadas de TRAIN (val nunca se toca)")
    X = build_matrix(ds, ids, use_cache=not args.quick)
    y = np.array([ds.record(i).label for i in ids])
    g = np.array([groups[i] for i in ids])

    fp = make_fit_predict(ds, X, ids, y)
    real = nested_cv(ds, proto, fp, groups)
    auc_real = auc(real.y_true, real.y_score)
    print(f"\n  AUC real       {auc_real:.4f}   (OOF sobre {len(real.y_true)} llamadas)")

    # Grupos conservadores = un grupo por llamada: permutar "dentro del grupo" no permuta
    # nada. Se permuta globalmente y se dice, en vez de reportar un test que no corrio.
    conservative = len({groups[i] for i in ids}) == len(ids)
    if conservative:
        y_perm = np.random.default_rng(20260912).permutation(y)
        print("  ! grupos conservadores (1 llamada = 1 grupo): permutacion GLOBAL")
    else:
        y_perm = permute_labels_within_groups(y, g)

    fp_perm = make_fit_predict(ds, X, ids, y_perm)
    perm = nested_cv(ds, proto, fp_perm, groups)
    y_true_perm = np.array([y_perm[list(ids).index(i)] for i in perm.ids])
    auc_perm = auc(y_true_perm, perm.y_score)
    print(f"  AUC permutado  {auc_perm:.4f}   (deberia ser ~0.5)")

    print()
    if not np.isfinite(auc_perm):
        print("INDETERMINADO: el AUC permutado no es finito.")
        return 2
    if auc_perm > args.tol:
        print(f"FALLA: {auc_perm:.4f} > {args.tol}. Hay fuga — algo se ajusta fuera de su "
              "fold. No sigas construyendo encima hasta entender que.")
        return 1
    if abs(auc_real - 0.5) < 0.05:
        print(f"INCONCLUSO: el AUC real ({auc_real:.4f}) tambien esta en el azar. El test "
              "no dice nada del protocolo, solo que estas features no separan.")
        return 0
    print(f"PASA: el protocolo aguanta. Real {auc_real:.4f} vs permutado {auc_perm:.4f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
