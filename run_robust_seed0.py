"""Robustness sweep across corruption conditions, seed 0.

Loads a set of trained YOLO-family checkpoints (baseline, EMA variant, and the
DaYa dual-branch variants including the RGB-only control) and evaluates each
one against seven versions of the validation set: one clean and six corrupted
(low light, overexposure, shadow, color shift, blur, noise).

For each model x condition pair it runs Ultralytics' val() to get mAP50,
mAP50-95, precision and recall, and caches results incrementally into
robust_seed0.json so a crashed/interrupted run can resume without repeating
already-completed pairs. At the end it prints a summary table showing each
model's percent change in mAP50 relative to its own clean-condition score,
which is the actual robustness comparison across conditions.
"""
import sys
import io
import os
import json
import pathlib
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', errors='replace')
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'
HERE = pathlib.Path(__file__).resolve().parent

import torch

WEIGHTS = HERE / 'weights'

NAMES = ['Bercak-Cokelat-Sempit', 'Blast', 'Busuk-Bulir', 'Busuk-Pelepah',
         'Gosong-Palsu', 'Hawar-Daun', 'bercak-Cokelat', 'brown plant hopper',
         'green leaf hopper', 'leaf folder', 'rice leaf roller', 'ricebug', 'stem borer']

MODELS = [
    ('YOLO11n (base)', 'yolo11n_0.pt'),
    ('YOLO11n + EMA', 'ema_0.pt'),
    ('DaYa-RGB', 'rgb_0.pt'),
    ('DaYa-XYZ', 'xyz_0.pt'),
    ('DaYa-LAB', 'lab_0.pt'),
]
CONDS = ['clean', 'low_light', 'overexposure', 'shadow', 'color_shift', 'blur', 'noise']
OUTFILE = HERE / 'robust_seed0.json'


def yaml_for(cond):
    d = HERE / 'corrupt' / cond
    y = HERE / ('data_%s.yaml' % cond)
    lines = ['path: %s' % d.as_posix(), 'train: images', 'val: images', 'names:']
    lines += ['  %d: %s' % (i, n) for i, n in enumerate(NAMES)]
    y.write_text('\n'.join(lines) + '\n', encoding='utf8')
    return y


def main():
    from ultralytics import YOLO
    out = json.loads(OUTFILE.read_text(encoding='utf8')) if OUTFILE.exists() else {}

    for label, ckpt in MODELS:
        out.setdefault(label, {})
        for cond in CONDS:
            if cond in out[label]:
                continue
            model = YOLO(str(WEIGHTS / ckpt))
            r = model.val(data=str(yaml_for(cond)), split='val', imgsz=640,
                          batch=16, device=0, verbose=False, plots=False,
                          save_json=False, project=str(HERE / 'robust0out'),
                          name='%s_%s' % (ckpt[:-3], cond), exist_ok=True)
            b = r.box
            out[label][cond] = {'mAP50': float(b.map50), 'mAP50_95': float(b.map),
                                'P': float(b.mp), 'R': float(b.mr)}
            print('  %-16s %-13s mAP50 %.4f  mAP50-95 %.4f'
                  % (label, cond, b.map50, b.map))
            OUTFILE.write_text(json.dumps(out, indent=1), encoding='utf8')
            del model
            torch.cuda.empty_cache()

    print('\n=== mAP50 change from clean, percent (seed 0) ===')
    print('%-16s' % 'model' + ''.join('%13s' % c for c in CONDS[1:]))
    for label, _ in MODELS:
        base = out[label]['clean']['mAP50']
        print('%-16s' % label + ''.join(
            '%12.1f%%' % (100.0 * (out[label][c]['mAP50'] - base) / base) for c in CONDS[1:]))


if __name__ == '__main__':
    main()