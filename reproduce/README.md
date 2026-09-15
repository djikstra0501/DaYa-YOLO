# Reproduction material

Everything behind the numbers in the paper: the per-seed outputs, the scripts
that turn them into the printed tables, the split manifests, the duplicate
detection, and the device records.

## Evaluation protocol

Every accuracy figure in the paper was measured the same way.

| setting | value |
| --- | --- |
| split | `test` |
| image size | 640 |
| batch | 16 |
| confidence threshold | 0.01 |
| NMS IoU threshold | 0.2 |
| maximum detections | 300 |
| framework | `ultralytics` 8.3.202, vendored in this repository |

## Which file backs which table

| Paper | File | Produced by |
| --- | --- | --- |
| Table 5, benchmark | `results/per_seed/table05_benchmark.json` | `scripts/eval_benchmark.py` |
| Table 6, paired comparison | `results/table06_paired.tex` | `scripts/paired_stats.py` |
| Table 7, per class | `results/per_seed/table07_per_class.json` | `scripts/eval_benchmark.py --per-class` |
| Table 8, robustness | `results/per_seed/table08_robustness.json` | `scripts/eval_robustness.py` |
| Table 8, aggregates | `results/table08_summary.json` | `scripts/summarise.py --robustness` |
| Tables 9 and 10, device | `results/jetson/` | see that directory's README |
| Contamination audit | `data/duplicates.json` | `scripts/find_duplicates.py` |
| Contamination re-evaluation | `results/per_seed/leakage_excluded.json`, `results/leakage_summary.json` | `scripts/eval_benchmark.py` on the filtered split |
| Split composition | `data/split_manifest.csv` | `scripts/build_manifest.py` |
| Noise amplification, Section 4.4 | printed by `scripts/encoder_noise.py` | |

Every result file carries a `_meta` block naming the protocol, the checkpoints
the rows were measured from, and the table it backs.

## Which checkpoints the paper reports

`configs/checkpoints.json` is the authoritative map. For each architecture it
lists every checkpoint, the seed **read out of the checkpoint's own training
record** rather than inferred from the filename, the training date and an md5.

The short version:

- Every architecture rests on five independent training runs, at the shared
  seeds 0, 14, 42, 56 and 81. There are 55 checkpoints in total.

- The three DaYa arms, including the architecture-matched control, come from
  `weights/{rgb,xyz,lab}_{0,14,42,56,81}.pt`. These are one series, trained back
  to back on 2026-09-06 at the same five seeds. That is what makes the control a
  matched one.
- The reference architectures come from `weights/<family>_<seed>.pt` and were
  trained between April and August 2026.
- `weights/legacy/` holds an earlier set of the three DaYa arms. **No number in
  the paper comes from it.** It is kept so the two sets can be compared.

## Reproducing from scratch

```bash
pip install -e .                      # installs the vendored framework

# 1. point the dataset config at your copy of the split
#    reproduce/configs/rice13.yaml expects <path>/{train,valid,test}/{images,labels}

# 2. accuracy, Tables 5 and 7
python reproduce/scripts/eval_benchmark.py \
    --data reproduce/configs/rice13.yaml --split test --per-class \
    --out reproduce/results/per_seed/table05_benchmark.json

# 3. paired comparison, Table 6
python reproduce/scripts/paired_stats.py \
    --results reproduce/results/per_seed/table05_benchmark.json \
    --model "DaYa-LAB" --against "Base (YOLO11)" "Base + EMA" "DaYa-RGB"

# 4. robustness, Table 8
python reproduce/scripts/make_corruptions.py --src <test split> --out corrupt
python reproduce/scripts/eval_robustness.py --corrupt corrupt \
    --out reproduce/results/per_seed/table08_robustness.json
python reproduce/scripts/summarise.py \
    --results reproduce/results/per_seed/table08_robustness.json --robustness

# 5. contamination audit
python reproduce/scripts/find_duplicates.py --dataset <split root>
python reproduce/scripts/build_manifest.py --dataset <split root>
```

Run the scripts from the repository root; they resolve the repository from their
own location and take everything machine-specific on the command line.

## Training

The recipe is not restated anywhere. `scripts/dump_train_args.py` reads the
arguments each reported run was launched with out of the checkpoint itself and
writes them to `configs/train_args/`, one file per run, 55 in total.
`scripts/train.py` reads one of those files back and relaunches it.

```bash
python reproduce/scripts/dump_train_args.py
python reproduce/scripts/train.py --model "DaYa-LAB" --seed 0 \
    --data reproduce/configs/rice13.yaml --dry-run
```

Every reported run used two NVIDIA T4 GPUs. On different hardware the effective
batch differs, so a rerun will not land on the same numbers even with the seed
held fixed.

## Reference architectures, as implemented here

`configs/checkpoints.json` carries a `notes` field per architecture recording
where the module sits and how the implementation differs from the source
publication. Three differences are worth stating here as well.

- **Base + BRA** does not shift the neck indices after the attention layer is
  inserted, so its detection head reads the three concatenations rather than the
  C3k2 blocks that follow them, and one block has no consumer. The row is
  reported as measured and is a weaker statement about that module than the
  others are about theirs.
- **MTD-YOLO** carries the MobileNetV3 backbone and the C2f-T neck but not the
  DyHead detection head, one of the three components its authors credit.
- **YOLO-PEST** carries the ConvNeXt fusion blocks and the CoTAttention module,
  but not the random-crop occlusion augmentation its authors list first, because
  the augmentation pipeline belongs to the shared recipe.

These are architecture-normalized results. They answer what each architecture
contributes when the training recipe is held fixed. They are not published-method
results and should not be read as a verdict on any published method.
