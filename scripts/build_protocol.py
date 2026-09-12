"""Congela los grupos y los folds. Se corre UNA vez; despues se lee, no se recalcula.

Produce:
    configs/protocol/groups_v1.csv     call_id -> group_id   (determinista, versionado)
    configs/protocol/folds_v1.json     los folds externos ya materializados
    experiments/val_looks.csv          el contador de miradas a `val`, vacio

Por que congelar: si los grupos se recalculan por corrida, los resultados de tres personas
dejan de ser comparables — y nadie se da cuenta, porque cada tabla se ve razonable sola.

`--verify` re-deriva todo y compara con lo congelado sin escribir nada. Eso es lo que corre
CI: detecta que alguien cambio las llaves de union sin actualizar el archivo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from altur.dataset import Dataset
from altur.protocol import (
    PROTOCOL_VERSION,
    build_groups,
    freeze_groups,
    get_protocol,
    load_groups,
)

ROOT = Path(__file__).resolve().parent.parent
FOLDS_JSON = ROOT / "configs" / "protocol" / "folds_v1.json"
VAL_LOOKS = ROOT / "experiments" / "val_looks.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="comparar sin escribir")
    args = ap.parse_args()

    ds = Dataset(require_audio=False)
    ga = build_groups(ds)

    print(f"Dataset: {len(ds)} llamadas · fingerprint {ds.fingerprint()[:16]}")
    print(f"Grupos:  {ga.n_groups} · conservador={ga.conservative}")
    print(f"Llaves de union con datos: {list(ga.linking_keys_used) or '(ninguna)'}")
    for w in ga.warnings:
        print(f"  ! {w}")

    if args.verify:
        frozen = load_groups()
        if frozen != ga.group_of:
            diff = [k for k in ga.group_of if frozen.get(k) != ga.group_of[k]]
            print(f"\nDIFIEREN en {len(diff)} llamadas, p.ej. {diff[:3]}", file=sys.stderr)
            print("Los grupos congelados ya no coinciden con los derivados. Si el cambio "
                  "es intencional, abre groups_v2: NO sobrescribas v1 con folds corridos.",
                  file=sys.stderr)
            return 1
        print("\nLos grupos congelados coinciden con los derivados.")
        return 0

    path = freeze_groups(ga)
    print(f"\nescrito {path.relative_to(ROOT)}")

    folds_out: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_fingerprint": ds.fingerprint(),
        "conservative_groups": ga.conservative,
        "group_warnings": list(ga.warnings),
        "protocols": {},
    }
    for name in ("official_v1", "pooled5_v1"):
        proto = get_protocol(name, ds)
        folds = proto.outer_folds(ds, ga.group_of)
        folds_out["protocols"][name] = {  # type: ignore[index]
            "n_fit": len(proto.fit_ids),
            "n_judge": len(proto.judge_ids),
            "val_contaminated": proto.val_contaminated,
            "seed": proto.seed,
            "folds": [{"train": list(tr), "test": list(te)} for tr, te in folds],
        }
        print(f"\n{name}: fit={len(proto.fit_ids)} juez={len(proto.judge_ids)}")
        for k, (tr, te) in enumerate(folds):
            syn = sum(ds.record(i).label == 1 for i in te)
            print(f"  fold {k}: train={len(tr):4d}  test={len(te):3d}  "
                  f"sinteticas en test={syn:3d} ({syn / len(te):.1%})")

    FOLDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FOLDS_JSON.write_text(json.dumps(folds_out, indent=2), encoding="utf-8")
    print(f"\nescrito {FOLDS_JSON.relative_to(ROOT)}")

    if not VAL_LOOKS.exists():
        VAL_LOOKS.parent.mkdir(parents=True, exist_ok=True)
        VAL_LOOKS.write_text("timestamp,run_id,who,reason\n", encoding="utf-8")
        print(f"escrito {VAL_LOOKS.relative_to(ROOT)} (vacio)")
    else:
        n = len(VAL_LOOKS.read_text(encoding="utf-8").strip().splitlines()) - 1
        print(f"{VAL_LOOKS.relative_to(ROOT)}: {n} miradas a val hasta ahora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
