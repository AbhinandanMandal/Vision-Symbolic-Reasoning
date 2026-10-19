# Bridging Pretrained Vision and Symbolic Reasoning Through Intermediate-Layer Grounding

**Symbolic Grounding Transfer (SGT)** — official repository for our paper accepted at **ACCV 2026**.

> 🚧 **Code release coming soon.** The code for this work (SGT decoder, YOLOv5s baseline, ELVIS/NEUMANN evaluation pipeline, calibration, and analysis scripts) will be uploaded and made publicly available soon. Watch or star this repository to be notified.

---

## Overview

Neuro-symbolic visual reasoning depends on the interface that converts observations into symbolic evidence. Existing pipelines rely on task-specific detectors (e.g., Mask R-CNN, Slot Attention), which can discard information about higher-order perceptual organization such as contours, scale, and spatial arrangement, exactly the properties Gestalt principles describe.

We ask whether **pretrained vision representations form a stronger perception-reasoning interface**. Using the [ELVIS](https://github.com/) Gestalt benchmark and the few-shot **NEUMANN** differentiable reasoner, we introduce **Symbolic Grounding Transfer (SGT)**: a decoder that maps intermediate **InternViT** representations into NEUMANN's object-centric symbolic interface, **leaving the reasoner unchanged**.

### Research questions

1. Do pretrained representations improve symbolic grounding over a task-specific detector?
2. Does representation depth matter?
3. Does the decision rule used to convert reasoning scores into predictions matter as much as the representation itself?

### Key findings

- **Intermediate layers ground better than the final layer.** Layer 16 of InternViT-300M-448px gives the best grounding score (0.545 vs. 0.400 for the final layer), measured by matched attribute accuracy and independent of any decision threshold.
- **Under the fixed 0.5 threshold, the advantage does not appear.** That threshold is inherited from the detector-based frontend and is poorly aligned with the new representations.
- **Training-only calibration reverses the ranking.** SGT-Layer reaches **0.577 F1 / 0.587 accuracy** vs. **0.493 / 0.543** for YOLO (paired Wilcoxon signed-rank, *p* = 0.018).
- **Largest gains on global-structure principles:** continuity, symmetry, and closure.
- Effective perception-reasoning transfer depends **jointly on representation depth and decision calibration**.

---

## Method

```
Image I → visual frontend → Z → symbolic grounding → V → NEUMANN → P(y = 1)
```

- **Common interface.** Every frontend is converted into object slots `Z_i = [b_i, c_i, s_i, o_i]` (normalized box, color logits, shape logits, objectness). Slot budgets are principle-specific: 16 slots for proximity, similarity, and continuity; 32 for closure and symmetry.
- **YOLO baseline.** YOLOv5s trained on pattern-free synthetic scenes (3,000 train / 1,000 val images, 20 epochs), following the original ELVIS baseline design at smaller scale.
- **SGT decoder.** A frozen InternViT-300M-448px backbone with a learned query-attention slot decoder trained with Hungarian matching on 2,000 pattern-free scenes. Loss: `L_ground = L_box + L_color + L_shape + L_obj`. AdamW, 12 epochs, lr 3e-4, batch size 8, 224×224 inputs.
- **Layer selection.** The same decoder is applied to layers {24, 20, 16, 12, 8, 4}, ranked by
  `S_ground = (A_color + A_shape) / 2 − E_box`.
- **LoRA.** Ranks {0, 4, 8, 16} on upper layers were swept; the sweep selected r = 0, so the reported SGT-Layer uses a **fully frozen backbone**.
- **Reasoning.** NEUMANN, the logical language, candidate clauses, task splits, training schedule, and evaluation are held fixed. Two decision protocols are reported: the original fixed threshold (ŷ = 1[P > 0.5]) and a **training-only calibrated threshold** selected per task from the six training examples and applied unchanged at test time.

---

## Experimental setup

- **Benchmark:** ELVIS, NM/3 few-shot binary classification (3 positive + 3 negative training examples, held-out test set of equal size) across five Gestalt principles.
- **Subset:** 125-task development subset (25 tasks per principle) out of 933 tasks.
- **Hardware:** single laptop with an RTX 5070 GPU, AMD Ryzen 9 8940HX, 24 GB RAM. NEUMANN runs on CPU (its sparse-scatter backend has no build for current CUDA versions).
- **Seeds:** single seed.

---

## Results

### Grounding quality across InternViT depth (no decision threshold)

| Block | Color Acc. | Shape Acc. | Box L1 ↓ | Grounding |
|-------|-----------:|-----------:|---------:|----------:|
| 24 (final) | 0.462 | 0.701 | 0.1817 | 0.400 |
| 20 | **0.613** | 0.691 | 0.1832 | 0.469 |
| **16** | 0.588 | **0.857** | **0.1777** | **0.545** |
| 12 | 0.559 | 0.854 | 0.1805 | 0.526 |
| 8 | 0.540 | 0.757 | 0.1849 | 0.464 |
| 4 | 0.594 | 0.604 | 0.1855 | 0.414 |

### Fixed threshold (0.5)

| Frontend | Accuracy | F1 | Precision | Recall |
|----------|---------:|---:|----------:|-------:|
| YOLO | 0.521 | 0.373 | 0.386 | 0.411 |
| Frozen InternViT | 0.515 | 0.213 | 0.239 | 0.219 |
| SGT-Layer | 0.516 | 0.159 | 0.206 | 0.149 |

### Training-only calibrated threshold

| Frontend | Accuracy | F1 | Precision | Recall |
|----------|---------:|---:|----------:|-------:|
| YOLO | 0.543 | 0.493 | 0.458 | 0.600 |
| Frozen InternViT | 0.569 | 0.549 | 0.504 | 0.661 |
| **SGT-Layer** | **0.587** | **0.577** | **0.546** | **0.675** |

SGT-Layer vs. YOLO: +0.084 absolute F1 (~17% relative), paired Wilcoxon *p* = 0.018 over 125 matched tasks. Frozen InternViT vs. YOLO: +0.056 F1, *p* = 0.097 (not significant).

### Calibrated F1 by Gestalt principle

| Frontend | Proximity | Similarity | Closure | Symmetry | Continuity |
|----------|----------:|-----------:|--------:|---------:|-----------:|
| YOLO | 0.599 | **0.567** | 0.494 | 0.422 | 0.382 |
| Frozen InternViT | **0.619** | 0.483 | 0.564 | 0.497 | 0.581 |
| SGT-Layer | 0.597 | 0.502 | **0.596** | **0.553** | **0.635** |

---

## Limitations

These results are **preliminary**: 125 of 933 ELVIS tasks, a single seed, quantized colors, fixed slot budgets, and a frozen backbone. We do not establish *why* continuity, symmetry, and closure benefit most; full-benchmark, multi-seed, and ablation studies are left to future work. Our YOLO reproduction is a sanity check against the published ELVIS NM/3 baseline, not an exact numerical replication.

---

## Code release

The following will be released publicly soon:

- [ ] SGT slot decoder and training code
- [ ] YOLOv5s baseline training on pattern-free synthetic scenes
- [ ] InternViT layer sweep and grounding-score evaluation
- [ ] ELVIS / NEUMANN integration and evaluation pipeline
- [ ] Training-only threshold calibration
- [ ] Scripts to reproduce all tables and statistical tests

Repository structure, installation instructions, and usage examples will be added at release.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{sgt2026,
  title     = {Bridging Pretrained Vision and Symbolic Reasoning Through Intermediate-Layer Grounding},
  author    = {TODO},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

This work builds on [ELVIS / Gestalt Vision](https://github.com/), [NEUMANN](https://github.com/), [InternVL / InternViT](https://github.com/OpenGVLab/InternVL), and [YOLOv5](https://github.com/ultralytics/yolov5).

## License

TBD at code release.
