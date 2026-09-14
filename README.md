# DaYa-YOLO

Dual-branch rice pest and disease detection with a frozen colorimetric branch,
built for edge deployment on the NVIDIA Jetson Nano.

This repository accompanies *DaYa-YOLO: Substituting a Frozen Colorimetric Prior
for Learned Attention in Edge-Deployed Rice Pest and Disease Detection*,
submitted to the International Journal of Intelligent Engineering and Systems.

Everything behind the numbers in the paper is in [`reproduce/`](reproduce/):
per-seed outputs, the scripts that turn them into the printed tables, the split
manifests, the contamination audit and the device records. Start with
[`reproduce/README.md`](reproduce/README.md).

---

## What this repository is

A fork of [Ultralytics](https://github.com/ultralytics/ultralytics) with a
dual-branch architecture added. Most of the code here is unmodified Ultralytics.

DaYa-YOLO runs an unmodified YOLO11 RGB backbone in parallel with a **Chromatic
Feature Encoder**, a fixed sRGB to CIE XYZ to CIELAB transform with a learnable
per-channel scaling applied downstream of it, and fuses the two branches at the
P3, P4 and P5 detection scales.

What is fixed and what is learned matters for reading the paper. The colour
conversion itself carries no learned parameters. Twelve learnable values, a
three-element channel scaling and a three by three projection initialised to the
identity, sit **after** the conversion, so the invariance the transform provides
holds at every point in training rather than being fitted.

Three arms share that branch and differ only in what the branch receives:

| Arm | Branch input | Role |
| --- | --- | --- |
| **DaYa-LAB** | sRGB to XYZ to CIELAB | proposed |
| **DaYa-XYZ** | sRGB to XYZ | proposed |
| **DaYa-RGB** | the image unchanged | architecture-matched control |

The control is what makes the comparison interpretable: it isolates what the
added branch does from what the colour transform inside it does.

## What the paper concludes

Over five shared seeds none of the eleven architectures separates from stock
YOLO11 on mean accuracy, and the control is indistinguishable from both transform
arms. The colour pathway is therefore not what the dual-branch design is doing.
Where the architectures do separate is deployment: eight of ten produced a
working TensorRT engine on the Jetson Nano and two, one of them an attention
module, produced none.

## Installation

```bash
git clone https://github.com/djikstra0501/YOLOv11-ECA-SAM.git
cd YOLOv11-ECA-SAM
pip install -e .
```

## Usage

```python
from ultralytics import YOLO

model = YOLO("weights/lab_0.pt")          # a reported checkpoint
model.val(data="reproduce/configs/rice13.yaml", split="test",
          imgsz=640, batch=16, conf=0.01, iou=0.2)
```

Building a variant from scratch:

```python
model = YOLO("ultralytics/cfg/models/11/yolov11-color.yaml")
```

Exporting for the Jetson Nano:

```python
model.export(format="engine", half=True, imgsz=640)
```

## Weights

`weights/` holds 53 checkpoints across eleven architectures, named
`<family>_<seed>.pt`.

**`reproduce/configs/checkpoints.json` is the authoritative map** from every
reported number to the file that produced it. For each checkpoint it records the
seed read out of the checkpoint's own training record, the training date and an
md5.

- The three DaYa arms come from `{rgb,xyz,lab}_{0,14,42,56,81}.pt`, one series
  trained back to back on 2026-09-06 at the same five seeds.
- The reference architectures were trained between April and August 2026.
- `weights/legacy/` holds an earlier set of the three DaYa arms. **No number in
  the paper comes from it.** It is kept so the two sets can be compared.

## Evaluation protocol

Test split, image size 640, batch 16, confidence threshold 0.01, NMS IoU
threshold 0.2, at most 300 detections per image. Both thresholds were held
identical for every architecture and every table reporting accuracy.

## Modified files

Everything not listed here is unmodified Ultralytics.

**Modules** in `ultralytics/nn/modules/conv.py`: `ChromaticFeatureEncoder`,
`RGBIdentityEncoder`, `LSKA`, `SEAtt`, `SCM`, `CCS`, `GSConv`, `GnConv`.
Reference modules used by the comparison architectures live alongside them and
are registered in `ultralytics/nn/tasks.py`.

**Model configurations** in `ultralytics/cfg/models/`:

| Path | Architecture |
| --- | --- |
| `11/yolov11-color.yaml` | DaYa-LAB, DaYa-XYZ, DaYa-RGB |
| `11/yolov11-EMA-REF.yaml` | Base + EMA |
| `11/yolov11-LSKA-REF.yaml` | Base + LSKA |
| `11/yolov11-BRA-REF.yaml` | Base + BRA |
| `v5/yolov5-PEST.yaml` | YOLO-PEST |
| `v8/yolov8n-MTD.yaml` | MTD-YOLO |

A naming note for anyone loading an older checkpoint: the encoder was once called
`SpectralFeatureEncoder` and is now `ChromaticFeatureEncoder`. The reproduction
scripts register the old name as an alias so either loads.

## Repository layout

```
reproduce/    per-seed results, scripts, configs, device records
weights/      reported checkpoints, plus legacy/ for the earlier DaYa set
tools/        development utilities, not needed to reproduce the paper
assets/       figures and field photographs
ultralytics/  the vendored framework
```

`tools/` is kept for provenance. Several of those scripts reference files that
are not in this repository and are not maintained; the reproduction path is
`reproduce/`.

## Dataset

Thirteen classes of rice pests and diseases, collected across four sessions at a
single site under uncontrolled outdoor light. `reproduce/configs/rice13.yaml`
gives the class order, which is the index order the checkpoints were trained
against; changing it silently invalidates every reported number.

`reproduce/data/split_manifest.csv` lists every file in every split with the
source identity its name encodes and its instance count.
`reproduce/data/duplicates.json` is the contamination audit described in the
paper.

## Citation

```bibtex
@article{dananjaya_dayayolo,
  title  = {DaYa-YOLO: Substituting a Frozen Colorimetric Prior for Learned
            Attention in Edge-Deployed Rice Pest and Disease Detection},
  author = {Dananjaya, I Kadek Dipastra Arka and others},
  note   = {Under review, International Journal of Intelligent Engineering and
            Systems},
  year   = {2026}
}
```

## License

AGPL-3.0, inherited from Ultralytics. See [LICENSE](LICENSE).

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
