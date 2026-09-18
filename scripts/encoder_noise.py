"""Re-measure how far the chromatic encoder moves under sensor noise.

Section 4.4 quotes a ratio of 2.79 for the CIELAB pathway and 1.11 for the
tristimulus one, measured on the earlier checkpoints. The reported models are
now the September series, so the measurement is repeated on those.

For each image the relative displacement of a module's output is
||f(x+n) - f(x)|| / ||f(x)||. The figure reported is that quantity for the
chromatic encoder divided by the same quantity for the first convolution of the
RGB backbone, so a value above one means the encoder moves further than the
backbone stem does under the same perturbation.
"""
import glob
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image

import argparse

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_here, ".."))
WORK = REPO
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--images", required=True, help="directory of test images")
_a = _p.parse_args()

SRC = _a.images
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"

N_IMAGES = 40
SIGMA = 14.0  # the same standard deviation the noise condition applies

CASES = [
    ("DaYa-LAB", os.path.join(REPO, "weights", "lab_0.pt")),
    ("DaYa-XYZ", os.path.join(REPO, "weights", "xyz_0.pt")),
]


def rel_shift(mod, a, b):
    with torch.no_grad():
        fa, fb = mod(a), mod(b)
    return (torch.norm(fb - fa) / torch.norm(fa)).item()


def main():
    os.chdir(WORK)
    from ultralytics import YOLO
    from ultralytics.nn.modules.conv import Conv

    paths = sorted(glob.glob(os.path.join(SRC, "*")))[:N_IMAGES]
    rng = np.random.default_rng(0)
    batch = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((640, 640))
        batch.append(np.asarray(im).astype(np.float32))
    clean = np.stack(batch)
    noisy = np.clip(clean + rng.normal(0, SIGMA, clean.shape), 0, 255)

    for label, path in CASES:
        if not os.path.exists(path):
            print("%-10s MISSING %s" % (label, path))
            continue
        model = YOLO(path).model.eval().cuda()
        enc = None
        for m in model.modules():
            if type(m).__name__ in ("ChromaticFeatureEncoder", "SpectralFeatureEncoder"):
                enc = m
                break
        stem = next(m for m in model.modules() if isinstance(m, Conv))
        if enc is None:
            print("%-10s no chromatic encoder found" % label)
            continue

        ratios = []
        for i in range(len(paths)):
            a = torch.from_numpy(clean[i] / 255.0).permute(2, 0, 1)[None].cuda()
            b = torch.from_numpy(noisy[i] / 255.0).permute(2, 0, 1).float()[None].cuda()
            e = rel_shift(enc, a, b)
            s = rel_shift(stem, a, b)
            if s > 0:
                ratios.append(e / s)
        print("%-10s ratio %.2f  (median %.2f, n=%d, sigma=%.0f/255)"
              % (label, float(np.mean(ratios)), float(np.median(ratios)),
                 len(ratios), SIGMA), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
