"""Shared helpers for the reproduction scripts.

Nothing here holds an absolute path. Every script resolves the repository from
its own location and takes the dataset location on the command line, so the
scripts run wherever the repository is cloned.
"""
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "reproduce" / "configs" / "checkpoints.json"

# Evaluation protocol. Every accuracy figure in the paper uses these values.
PROTOCOL = dict(imgsz=640, batch=16, conf=0.01, iou=0.2, max_det=300)


def setup():
    """Put the vendored package first on the path and silence progress output."""
    sys.path.insert(0, str(ROOT))
    warnings.filterwarnings("ignore")
    os.environ.setdefault("YOLO_VERBOSE", "False")


def registry():
    return json.loads(REGISTRY.read_text(encoding="utf8"))


def checkpoints(labels=None):
    """Yield (label, seed, absolute path) for the requested architectures."""
    reg = registry()["architectures"]
    for label, entry in reg.items():
        if labels and label not in labels:
            continue
        for c in entry["checkpoints"]:
            yield label, c["seed"], ROOT / c["file"]


def load_results(path):
    return json.loads(Path(path).read_text(encoding="utf8")) if Path(path).exists() else {}


def save_results(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf8")


def graft_legacy_modules():
    """Checkpoints written before the encoder was renamed reference the old class
    name. Registering an alias lets them unpickle against the current package."""
    import torch.nn as nn
    import ultralytics.nn.modules.conv as conv
    if hasattr(conv, "ChromaticFeatureEncoder") and not hasattr(conv, "SpectralFeatureEncoder"):
        conv.SpectralFeatureEncoder = conv.ChromaticFeatureEncoder
    return nn
