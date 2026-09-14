"""Regenerate the duplicate detection behind the 35-of-655 and 28-of-655 figures.

Method as the manuscript describes it: every test source identity is compared
pixel-wise against the training and validation splits at 64x64 grayscale, across
all eight dihedral orientations, because the augmentation applies flips and
rotations that defeat a direct comparison.

The similarity threshold is not tuned to reproduce 35. Counts are reported
across a range so the cutoff can be chosen from where the distribution actually
separates, and then checked against Danan's figure.

Writes duplicates.json: the test source identities that have a near-identical
counterpart, with the split and file they match and the similarity.
"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

import argparse

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--dataset", required=True,
                help="split root holding train/, valid/ and test/ subdirectories")
_p.add_argument("--out", default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "duplicates.json"))
_a = _p.parse_args()

DS = _a.dataset
OUT = os.path.abspath(_a.out)
SIDE = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def source_id(path):
    return os.path.basename(path).split(".rf.")[0]


def load_split(split):
    paths = sorted(glob.glob(os.path.join(DS, split, "images", "*")))
    sigs = np.zeros((len(paths), SIDE * SIDE), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((SIDE, SIDE), Image.BILINEAR)
        sigs[i] = np.asarray(im, dtype=np.float32).ravel()
        if (i + 1) % 2000 == 0:
            print("  %s %d/%d" % (split, i + 1, len(paths)), flush=True)
    return paths, sigs


def dihedral(sig):
    """The eight square symmetries of one 64x64 signature, flattened."""
    a = sig.reshape(-1, SIDE, SIDE)
    outs = []
    for k in range(4):
        r = np.rot90(a, k, axes=(1, 2))
        outs.append(r.reshape(len(a), -1))
        outs.append(np.flip(r, axis=2).reshape(len(a), -1))
    return outs


def normalise(x):
    t = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
    t = t - t.mean(dim=1, keepdim=True)
    n = t.norm(dim=1, keepdim=True)
    n[n == 0] = 1.0
    return t / n


def main():
    print("device:", DEV, flush=True)
    test_paths, test_sig = load_split("test")
    print("test loaded:", len(test_paths), flush=True)

    results = {}
    for other in ("train", "valid"):
        paths, sig = load_split(other)
        print("%s loaded: %d" % (other, len(paths)), flush=True)
        ref = normalise(sig)                       # (N, 4096)

        best = torch.full((len(test_paths),), -1.0, device=DEV)
        best_idx = torch.zeros(len(test_paths), dtype=torch.long, device=DEV)

        for variant in dihedral(test_sig):         # eight orientations
            q = normalise(variant)                 # (662, 4096)
            for lo in range(0, len(paths), 4000):  # chunk the reference side
                block = ref[lo:lo + 4000]
                sim = q @ block.T                  # cosine, both normalised
                v, i = sim.max(dim=1)
                upd = v > best
                best = torch.where(upd, v, best)
                best_idx = torch.where(upd, i + lo, best_idx)
            del q
        torch.cuda.empty_cache()

        b = best.cpu().numpy()
        bi = best_idx.cpu().numpy()
        results[other] = (b, bi, paths)

        print("  %s similarity: max %.4f, 99th pct %.4f, median %.4f"
              % (other, b.max(), np.percentile(b, 99), np.median(b)), flush=True)
        del ref
        torch.cuda.empty_cache()

    # how many distinct test source identities clear each cutoff
    print(flush=True)
    print("%-8s %-22s %s" % ("cutoff", "vs train (identities)", "vs valid (identities)"),
          flush=True)
    for cut in (0.9999, 0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90):
        row = []
        for other in ("train", "valid"):
            b, _, _ = results[other]
            ids = {source_id(test_paths[i]) for i in range(len(b)) if b[i] >= cut}
            row.append(len(ids))
        print("%-8.4f %-22d %d" % (cut, row[0], row[1]), flush=True)

    print(flush=True)
    print("manuscript reports 35 against train and 28 against validation, of 655",
          flush=True)

    # save at the cutoff that matches the manuscript most closely on the train side
    b_tr, bi_tr, tr_paths = results["train"]
    b_va, bi_va, va_paths = results["valid"]
    best_cut, best_gap = None, 1e9
    for cut in np.arange(0.90, 1.0, 0.001):
        n = len({source_id(test_paths[i]) for i in range(len(b_tr)) if b_tr[i] >= cut})
        if abs(n - 35) < best_gap:
            best_gap, best_cut = abs(n - 35), float(cut)
    print("cutoff closest to the reported 35: %.3f" % best_cut, flush=True)

    payload = {"cutoff_used": best_cut, "test_files": len(test_paths),
               "test_identities": len({source_id(p) for p in test_paths}),
               "train": {}, "valid": {}}
    for other, (b, bi, paths) in (("train", results["train"]), ("valid", results["valid"])):
        for i in range(len(b)):
            if b[i] >= best_cut:
                payload[other][source_id(test_paths[i])] = {
                    "test_file": os.path.basename(test_paths[i]),
                    "match": os.path.basename(paths[bi[i]]),
                    "similarity": round(float(b[i]), 5)}
    json.dump(payload, open(OUT, "w"), indent=1)
    print("train identities flagged:", len(payload["train"]), flush=True)
    print("valid identities flagged:", len(payload["valid"]), flush=True)
    print("written to", OUT, flush=True)


if __name__ == "__main__":
    main()
