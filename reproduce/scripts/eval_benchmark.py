"""Score every checkpoint in the registry on one split.

This produces the per-seed records behind Tables 5 and 7. Aggregate rows in the
paper are the mean and sample standard deviation over the seeds reported here.

    python reproduce/scripts/eval_benchmark.py --data reproduce/configs/rice13.yaml \
        --split test --out reproduce/results/per_seed/table05_benchmark.json
"""
import argparse

from _common import PROTOCOL, checkpoints, load_results, save_results, setup, graft_legacy_modules


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dataset YAML")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", required=True, help="where to write the per-seed JSON")
    p.add_argument("--device", default="0")
    p.add_argument("--models", nargs="*", default=None,
                   help="architecture labels to run; default is all of them")
    p.add_argument("--per-class", action="store_true",
                   help="also record per-class precision and mAP50")
    return p.parse_args()


def main():
    a = parse_args()
    setup()
    import torch
    graft_legacy_modules()
    from ultralytics import YOLO

    out = load_results(a.out)
    results = out.get("results", {})
    for label, seed, path in checkpoints(a.models):
        results.setdefault(label, {})
        if str(seed) in results[label]:
            continue
        if not path.exists():
            print("%-16s seed %-3s missing %s" % (label, seed, path))
            continue
        m = YOLO(str(path))
        r = m.val(data=a.data, split=a.split, device=a.device, workers=0,
                  verbose=False, plots=False, save_json=False, **PROTOCOL)
        rec = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
               "P": float(r.box.mp), "R": float(r.box.mr)}
        if a.per_class:
            rec["per_class_P"] = [float(x) for x in r.box.p]
            rec["per_class_mAP50"] = [float(x) for x in r.box.ap50]
        results[label][str(seed)] = rec
        out["results"] = results
        save_results(a.out, out)
        print("%-16s seed %-3s mAP50 %.4f  mAP50-95 %.4f" % (label, seed, r.box.map50, r.box.map))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
