"""Extract the training configuration recorded inside each checkpoint.

Ultralytics stores the arguments a run was launched with in the checkpoint
itself. This reads them back without building the model, so the recipe,
hyperparameters and random seed of every reported run come from the files rather
than from a description of them.

    python scripts/dump_train_args.py
"""
import argparse
import json
import pickle
import zipfile
from pathlib import Path

from _common import ROOT, checkpoints


class _Stub:
    """Stands in for any class referenced by the pickle, so nothing is imported."""

    def __init__(self, *args, **kwargs):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Stub

    def persistent_load(self, pid):
        return None


def read_meta(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        with z.open(name) as fh:
            return _Unpickler(fh).load()


def slug(label):
    out = label.lower().replace(" + ", "_plus_").replace(" ", "_")
    return out.replace("(", "").replace(")", "")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "configs" / "train_args"))
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for label, seed, path in checkpoints():
        if not path.exists():
            print("%-16s seed %-3s missing" % (label, seed))
            continue
        meta = read_meta(path)
        args = dict(meta.get("train_args") or {})
        args["_recorded_date"] = str(meta.get("date"))
        args["_source_file"] = path.name
        name = "%s_seed%s.json" % (slug(label), seed)
        (out / name).write_text(json.dumps(args, indent=1, default=str, sort_keys=True),
                                encoding="utf8")
        written += 1
    print("wrote %d configurations to %s" % (written, out))


if __name__ == "__main__":
    main()
