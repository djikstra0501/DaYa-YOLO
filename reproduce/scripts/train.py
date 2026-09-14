"""Retrain one architecture under the recipe the reported runs actually used.

The recipe is not restated here. It is read from the configuration dumped out of
the checkpoints themselves by dump_train_args.py, so what this script launches is
what the reported run was launched with, minus the paths, which are taken from
the command line because they belong to the machine that trained it.

    python reproduce/scripts/dump_train_args.py
    python reproduce/scripts/train.py --model "DaYa-LAB" --seed 0 \
        --data reproduce/configs/rice13.yaml

Every reported run used two NVIDIA T4 GPUs. On different hardware the effective
batch, and therefore the result, will differ even with the seed held fixed.
"""
import argparse
import json
from pathlib import Path

from _common import ROOT, registry, setup

# keys that describe where a run happened rather than how, so they are replaced
MACHINE_KEYS = {"model", "data", "project", "name", "save_dir", "device", "workers",
                "exist_ok", "resume", "pretrained"}


def slug(label):
    out = label.lower().replace(" + ", "_plus_").replace(" ", "_")
    return out.replace("(", "").replace(")", "")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="architecture label from checkpoints.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", required=True, help="dataset YAML")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/reproduce")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved configuration and exit")
    return p.parse_args()


def main():
    a = parse_args()
    cfg_path = ROOT / "reproduce" / "configs" / "train_args" / (
        "%s_seed%d.json" % (slug(a.model), a.seed))
    if not cfg_path.exists():
        raise SystemExit("no recorded configuration at %s; run dump_train_args.py first"
                         % cfg_path)
    recorded = json.loads(cfg_path.read_text(encoding="utf8"))
    cfg = {k: v for k, v in recorded.items()
           if not k.startswith("_") and k not in MACHINE_KEYS and v is not None}

    arch = registry()["architectures"][a.model]
    cfg.update(data=a.data, device=a.device, project=a.project,
               name="%s_seed%d" % (slug(a.model), a.seed), exist_ok=True)

    print("architecture : %s" % a.model)
    print("configuration: %s" % arch["config"])
    print("recorded on  : %s" % recorded.get("_recorded_date"))
    print("seed         : %s" % cfg.get("seed"))
    print("epochs       : %s" % cfg.get("epochs"))
    print("batch        : %s" % cfg.get("batch"))
    if a.dry_run:
        print(json.dumps(cfg, indent=1, sort_keys=True, default=str))
        return

    setup()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / arch["config"]) if Path(arch["config"]).suffix == ".yaml"
                 and (ROOT / arch["config"]).exists() else arch["config"])
    model.train(**cfg)


if __name__ == "__main__":
    main()
