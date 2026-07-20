# DaYa-YOLO

Dual-branch rice pest and disease detection with a physics-grounded chromatic
branch, optimized for edge deployment on the NVIDIA Jetson Nano.

This repository accompanies the paper *"[PAPER TITLE]"* ([VENUE], [YEAR]).

---

## What this repository is

This is a **fork of [Ultralytics](https://github.com/ultralytics/ultralytics)**
with a custom dual-branch architecture added. The overwhelming majority of the
code here is unmodified Ultralytics; our contribution is confined to the files
listed under [Modified files](#modified-files).

DaYa-YOLO runs an unmodified YOLO11 RGB backbone in parallel with a **Chromatic
Feature Encoder (CFE)** — a frozen, colorimetrically exact sRGB → CIE XYZ →
CIELAB transform with learnable per-channel scaling — and fuses the two branches
at the P3, P4, and P5 detection scales.

> **Naming note:** the CFE is implemented in code as `SpectralFeatureEncoder`.
> The paper refers to it as the Chromatic Feature Encoder (CFE). They are the
> same module.

Two variants are provided, selected by the encoder's mode argument:

| Variant       | CFE mode | Config |
|---------------|----------|--------|
| **DaYa-LAB**  | `"LAB"`  | `ultralytics/cfg/models/11/yolov11-color.yaml` |
| **DaYa-XYZ**  | `"XYZ"`  | `ultralytics/cfg/models/11/yolov11-color.yaml` |

---

## Modified files

The following files differ from upstream Ultralytics. Everything else is
unchanged.

<!-- TODO: confirm these paths against your actual diff before publishing -->
- `ultralytics/nn/modules/block.py` — `SpectralFeatureEncoder` (CFE) implementation
- `ultralytics/nn/modules/__init__.py` — module export
- `ultralytics/nn/tasks.py` — module registration for YAML parsing
- `ultralytics/cfg/models/11/yolov11-color.yaml` — DaYa-YOLO architecture
- `tools/` — training, evaluation, and analysis scripts (ours)

To see the exact diff against upstream:

```bash
git remote add upstream https://github.com/ultralytics/ultralytics.git
git fetch upstream
git diff upstream/main --stat
```

---

## Installation

```bash
git clone https://github.com/[USER]/[REPO].git
cd [REPO]
pip install -e .
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.8, as per upstream Ultralytics.

---

## Usage

### Training

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolov11-color.yaml")
model.train(data="your_dataset.yaml", epochs=..., imgsz=640, device=[0, 1])
```

Set the CFE mode (`"LAB"` or `"XYZ"`) in the YAML at the
`SpectralFeatureEncoder` layer before training. The mode is fixed at
construction time, not switchable at inference.

### Validation

```python
model = YOLO("weights/daya_lab_seed0.pt")
model.val(data="your_dataset.yaml", split="test")
```

### Inference

```python
results = model.predict("image.jpg", imgsz=640, conf=0.35)
results[0].show()
```

### Edge deployment (Jetson Nano)

```python
model.export(format="engine", half=True)  # TensorRT FP16
```

Our Jetson Nano software stack is a non-standard decoupled configuration
(JetPack 4.6.6 / L4T 32.7.6 with an upgraded Ubuntu 20.04 userspace); see the
paper's deployment section for the full version matrix.

---

## Pretrained weights

Trained weights are published under
[Releases]([RELEASES_URL]) rather than committed to the repository.

| File | Description |
|------|-------------|
| `daya_lab_seed{0,14,56}.pt` | DaYa-LAB, three training seeds |
| `daya_xyz_seed{0,14,56}.pt` | DaYa-XYZ, three training seeds |
| `yolo11n_seed{0,14,56}.pt`  | YOLO11n baseline, three training seeds |
| `ref_*.pt`                  | Reimplemented reference architectures (see below) |

---

## Reproducibility notes

**Seeds.** All reported results use seeds 0, 14, and 56. Metrics in the paper
are the mean and population standard deviation (n = 3) across these seeds.

**Training environment.** All models were trained on dual NVIDIA T4 GPUs
(Kaggle) under default Ultralytics settings.

**Reference architectures.** The `ref_*` weights are *our reimplementations* of
published architectures (YOLO-PEST, MTD-YOLO, and the EMA / LSKA / BRA attention
modules), built from each project's public code and trained here under a
**standardized configuration** — identical loss function, optimizer, and
schedule across all models — in order to isolate architectural differences from
variation in training recipe. Model-specific custom losses described in the
original papers were deliberately not reproduced. **These are not the original
authors' released weights and should not be read as a restatement of their
published results.**

**Not reported in the paper.** This repository also contains exploratory
configurations (ECA, SE, SAM, SCM, PGI, MCBAM, VoVGSCSP) that were investigated
during development but are not part of the reported study.

---

## Dataset

<!-- TODO: fill in or remove depending on what you're permitted to release -->
The dataset used in this work covers 13 rice pest and disease classes and was
compiled from field collection and an existing public dataset. See the paper for
composition and licensing details.

---

## Citation

If you use this work, please cite:

```bibtex
@article{[KEY],
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  doi     = {[DOI]}
}
```

---

## License

This project is a derivative work of Ultralytics YOLO and is distributed under
the **AGPL-3.0 License**, inherited from upstream. See [LICENSE](LICENSE).

---

## Acknowledgements

This work is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
We are grateful to the Ultralytics team for developing and openly releasing the
framework that made this research possible — the training, validation, export,
and deployment infrastructure used throughout this project is theirs, and
DaYa-YOLO would not exist without it.

```bibtex
@software{ultralytics_yolo,
  author  = {Jocher, Glenn and Qiu, Jing},
  title   = {Ultralytics {YOLO11}},
  year    = {2024},
  url     = {https://github.com/ultralytics/ultralytics}
}
```

We also thank our partner institutions for field data collection and
agronomic validation.