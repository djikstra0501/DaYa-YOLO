"""Write the split manifest Reviewer 1 asks for in comment 7.

One row per image: split, filename, the source identity the filename encodes,
the instance count, and whether that identity has a near-identical counterpart
in the training split according to the duplicate detection.
"""
import csv
import glob
import json
import os

import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--dataset", required=True,
                help="split root holding train/, valid/ and test/ subdirectories")
_p.add_argument("--duplicates", default=os.path.join(_here, "..", "data", "duplicates.json"))
_p.add_argument("--out", default=os.path.join(_here, "..", "data", "split_manifest.csv"))
_a = _p.parse_args()

DS = _a.dataset
DATA = os.path.dirname(os.path.abspath(_a.duplicates))
OUT = os.path.abspath(_a.out)

dup = json.load(open(os.path.join(DATA, "duplicates.json")))
flag_train = set(dup["train"])
flag_valid = set(dup["valid"])


def source_id(path):
    return os.path.basename(path).split(".rf.")[0]


rows = []
for split in ("train", "valid", "test"):
    for p in sorted(glob.glob(os.path.join(DS, split, "images", "*"))):
        name = os.path.basename(p)
        sid = source_id(p)
        lab = os.path.join(DS, split, "labels", os.path.splitext(name)[0] + ".txt")
        n = 0
        if os.path.exists(lab):
            n = sum(1 for l in open(lab) if l.strip())
        rows.append({
            "split": split,
            "file": name,
            "source_identity": sid,
            "instances": n,
            "near_duplicate_of_train": "yes" if (split == "test" and sid in flag_train) else "",
            "near_duplicate_of_valid": "yes" if (split == "test" and sid in flag_valid) else "",
        })

with open(OUT, "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

by_split = {}
for r in rows:
    s = by_split.setdefault(r["split"], [0, 0, set()])
    s[0] += 1
    s[1] += r["instances"]
    s[2].add(r["source_identity"])

print("%-7s %7s %10s %10s" % ("split", "files", "instances", "identities"))
for s in ("train", "valid", "test"):
    a = by_split[s]
    print("%-7s %7d %10d %10d" % (s, a[0], a[1], len(a[2])))
print()
print("test images flagged against train:",
      sum(1 for r in rows if r["near_duplicate_of_train"] == "yes"))
print("written to", OUT)
