"""Score checkpoints on the clean split and the six degraded ones.

This produces the per-seed records behind Table 8. Build the degraded sets first
with make_corruptions.py; the clean condition is scored on the re-encoded copy
that pass writes, so the change column isolates the degradation from the
re-encoding that accompanies it.

    python reproduce/scripts/make_corruptions.py --src <test split> --out corrupt
    python reproduce/scripts/eval_robustness.py --corrupt corrupt \
        --out reproduce/results/per_seed/table08_robustness.json
"""
import argparse
from pathlib import Path

from _common import (PROTOCOL, checkpoints, graft_legacy_modules, load_results,
                     save_results, setup)

CONDITIONS = ["clean", "low_light", "overexposure", "shadow", "color_shift",
              "blur", "noise"]
CLASSES = ["Bercak-Cokelat-Sempit", "Blast", "Busuk-Bulir", "Busuk-Pelepah",
           "Gosong-Palsu", "Hawar-Daun", "bercak-Cokelat", "brown plant hopper",
           "green leaf hopper", "leaf folder", "rice leaf roller", "ricebug",
           "stem borer"]
DEFAULT_MODELS = ["Base (YOLO11)", "Base + EMA", "DaYa-RGB", "DaYa-XYZ", "DaYa-LAB"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrupt", required=True,
                   help="directory holding one subdirectory per condition")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    return p.parse_args()


def yaml_for(root, cond, workdir):
    """Write a dataset YAML pointing at one condition directory."""
    path = Path(workdir) / ("data_%s.yaml" % cond)
    lines = ["path: %s" % (Path(root) / cond).resolve().as_posix(),
             "train: images", "val: images", "names:"]
    lines += ["  %d: %s" % (i, n) for i, n in enumerate(CLASSES)]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return str(path)


def main():
    a = parse_args()
    setup()
    import torch
    graft_legacy_modules()
    from ultralytics import YOLO

    workdir = Path(a.out).resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    yamls = {c: yaml_for(a.corrupt, c, workdir) for c in CONDITIONS}

    out = load_results(a.out)
    results = out.get("results", {})
    for label, seed, path in checkpoints(a.models):
        results.setdefault(label, {}).setdefault(str(seed), {})
        for cond in CONDITIONS:
            if cond in results[label][str(seed)]:
                continue
            m = YOLO(str(path))
            r = m.val(data=yamls[cond], split="val", device=a.device, workers=0,
                      verbose=False, plots=False, save_json=False, **PROTOCOL)
            results[label][str(seed)][cond] = {
                "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                "P": float(r.box.mp), "R": float(r.box.mr)}
            out["results"] = results
            save_results(a.out, out)
            del m
            torch.cuda.empty_cache()
        print("%-16s seed %-3s done" % (label, seed))


if __name__ == "__main__":
    main()
