"""Build illumination and degradation variants of the test split (revision point 8).

The reviewer asked whether the colorimetric branch actually buys robustness under
field lighting. That is a falsifiable prediction: CIELAB separates lightness from
the two chromatic axes, so a model reading L*a*b* should suffer less under a
change of illuminant colour than an RGB-only model. The colour_shift condition is
the direct test of that claim; the rest bound it.

Every corruption is deterministic, so the comparison across models is exact.
"""
import sys, io, glob, pathlib, shutil, argparse
import numpy as np
from PIL import Image, ImageFilter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
def parse_args():
    p = argparse.ArgumentParser(
        description='Build illumination/degradation variants of a test split.'
    )
    p.add_argument('--src', type=str, required=True,
                    help="Path to the source split root (contains images/ and labels/).")
    p.add_argument('--out', type=str, required=True,
                    help='Path to output folder for corrupted variants.')
    return p.parse_args()

args = parse_args()
SRC = pathlib.Path(args.src).resolve()
OUT = pathlib.Path(args.out).resolve()
IMGS = sorted(glob.glob(str(SRC / 'images' / '*')))
print('source images:', len(IMGS))


def gamma(a, g):
    return np.clip(((a / 255.0) ** g) * 255.0, 0, 255)


def f_clean(a, rng):
    return a


def f_low_light(a, rng):
    """Underexposure: gamma up, then a global gain below one."""
    return gamma(a, 2.2) * 0.55


def f_overexposure(a, rng):
    """Blown highlights: gain above one, clipped, plus gamma down."""
    return np.clip(gamma(a, 0.55) * 1.35, 0, 255)


def f_shadow(a, rng):
    """A soft canopy shadow across part of the frame, not a uniform dim."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    band = ((xx / w) * 1.4 + (yy / h) * 0.6)
    mask = 1.0 - 0.62 * np.clip(1.0 - np.abs(band - 1.0) * 1.6, 0, 1)
    return a * mask[..., None]


def f_color_shift(a, rng):
    """Illuminant change: warm the red channel, cool the blue one.
    This is the condition the CIELAB argument predicts it should resist."""
    gains = np.array([1.22, 1.0, 0.80])
    return np.clip(a * gains, 0, 255)


def f_blur(a, rng):
    im = Image.fromarray(a.astype('uint8'))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=2.0))).astype(float)


def f_noise(a, rng):
    return np.clip(a + rng.normal(0, 14.0, a.shape), 0, 255)


CONDS = {
    'clean': f_clean,
    'low_light': f_low_light,
    'overexposure': f_overexposure,
    'shadow': f_shadow,
    'color_shift': f_color_shift,
    'blur': f_blur,
    'noise': f_noise,
}

for name, fn in CONDS.items():
    d = OUT / name
    (d / 'images').mkdir(parents=True, exist_ok=True)
    (d / 'labels').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)          # deterministic per condition
    for p in IMGS:
        a = np.asarray(Image.open(p).convert('RGB')).astype(float)
        b = np.clip(fn(a, rng), 0, 255).astype('uint8')
        Image.fromarray(b).save(d / 'images' / pathlib.Path(p).name, quality=95)
        lab = pathlib.Path(str(p).replace('images', 'labels')).with_suffix('.txt')
        if lab.exists():
            shutil.copy(lab, d / 'labels' / lab.name)
    print('  built %-14s %d images' % (name, len(IMGS)))

print('done ->', OUT)
