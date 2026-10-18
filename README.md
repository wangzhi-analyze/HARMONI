# HARMONI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch_geometric-3C2179)](https://pyg.org/)

Variational graph autoencoder models for herbal medicine–drug interaction (HDI) prediction.
This repository contains only the four model variants requested for open-source release:

| Variant | Adaptive sample weighting (AS) | Relation modeling (RM) |
|---|:---:|:---:|
| `DVGAE` |  |  |
| `HARMONI-AS` | ✓ |  |
| `HARMONI-RM` |  | ✓ |
| `HARMONI` | ✓ | ✓ |

The repository contains source data required to run the examples, but intentionally does **not** contain checkpoints, predictions, logs, figures, benchmark tables, or any other generated run results.

## Repository layout

```text
HARMONI/
├── dataset/
│   ├── DHI.csv
│   ├── Drug_feature.csv
│   └── Herb_Finger_feature.csv
├── models.py       # DVGAE and the HARMONI AS/RM building blocks
├── train.py        # single training/evaluation entry point
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository and enter the directory:

```bash
git clone https://github.com/wangzhi-analyze/HARMONI.git
cd HARMONI
```

Python 3.8–3.11 is recommended. Install a PyTorch build appropriate for your CPU/GPU first, then install the remaining dependencies:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
```

For CUDA, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and then install `requirements.txt`.

## Run one model

Run from the repository root:

```bash
python train.py --model DVGAE --epochs 1000
```

Use `--model HARMONI-AS`, `--model HARMONI-RM`, or `--model HARMONI` for the other variants. A short smoke run is:

```bash
python train.py --model HARMONI --epochs 2 --channels 32 --cmw_warmup 0 --inference_repeats 1
```

By default, checkpoints are written under `outputs/`, which is ignored by Git. You can choose another location with `--checkpoint_path` and `--vnet_checkpoint_path`. Training metrics, predictions, and a JSON summary are opt-in through `--metrics_log_path`, `--prediction_path`, and `--result_json_path`.

Useful options include:

```text
--seed                 Random seed (default: 2025)
--epochs               Number of training epochs
--channels             Hidden graph embedding size (default: 128)
--feature_dim          Common feature dimension F0 after modality projection (default: 768)
--data_dir             Directory containing the three CSV input files
--split_path           Optional .npz split manifest with train/meta/val/test edges
--tune_threshold       Tune the evaluation threshold on the validation split
--validation_only      Evaluate on validation edges only
```

For dynamic weighting, the defaults use `rho = 0.2` and clip the normalized
weight `q_e` to `[0.5, 1.5]`. Therefore
`w_e = (1 - rho) + rho * q_e` is restricted to `[0.9, 1.1]`.

## Data format

`DHI.csv` must contain `Drug_ID` and `Food_Herb_ID` columns. The feature files use the entity identifier as the first (index) column and 768 numeric feature columns. Drug and herb identifiers must match the IDs in `DHI.csv`.

The MIT license applies to the source code in this repository. If you redistribute or reuse the bundled datasets, also check and follow the terms of the original dataset/feature providers.

The ablation study uses a matched `2 x 2` factorial design:

| Variant | Predictor positives | Meta positives | Edge weights | Graph refinement |
|---|---:|---|---|---|
| `DVGAE` | 5,405 inner-training pairs | Not used for updates | `w_e = 1` | Off |
| `HARMONI-AS` | 5,405 inner-training pairs | Update VNet only | Learned | Off |
| `HARMONI-RM` | 5,405 inner-training pairs | Not used for updates | `w_e = 1` | On |
| `HARMONI` | 5,405 inner-training pairs | Update VNet only | Learned | On |

For the bundled 7,508 positive pairs, cumulative 72%/80%/90% boundaries
produce 5,405 inner-training, 601 meta-learning, 751 validation, and 751 test
pairs. All four predictors are updated only from the same inner-training
pairs. The meta step never calls the predictor optimizer; when DW is enabled,
it updates only VNet. Its random-number state is isolated so matched DW and
non-DW runs use identical predictor negative samples, variational samples,
and Gumbel samples for the same seed.

Before graph encoding, the drug and herb inputs are aligned with two jointly
learned, modality-specific projections. Drug features are projected directly;
herb cluster-count features are first transformed with `log(1 + x)`. Each
projected row is L2-normalized, and the two blocks are then concatenated into
the common node-feature matrix. This corresponds to Eqs. (4)-(5) in the revised
method description.

The training objective implements Eq. (24): the (optionally dynamically
weighted) per-edge reconstruction losses are averaged over the training edge
batch, the two KL terms are scaled by `N / |B_tr|`, and the Laplacian loss is
added as `lambda_Lap * L_Lap`. Dynamic weights are detached during the model
update (`sg[w_e(phi)]`); gradients through them are retained only in the
bilevel VNet inner step used to learn `phi`.

## Reproducibility and outputs

The code seeds Python, NumPy, and PyTorch. Exact results can still vary across hardware and PyTorch/CUDA versions. Generated files are deliberately excluded from the repository; after a run, remove `outputs/` (or use a temporary output directory) if you want a clean working tree.

## Citation

If you find this work or codebase helpful in your research, please consider citing:

```bibtex
@article{harmoni2026,
  title={Variational Graph Autoencoders for Herbal Medicine--Drug Interaction Prediction},
  author={HARMONI Contributors},
  year={2026}
}
```

## Contributing and Issues

Contributions, suggestions, and issue reports are welcome! Please feel free to open an issue or pull request on [GitHub](https://github.com/wangzhi-analyze/HARMONI).

## License

Released under the MIT License. See [LICENSE](LICENSE).
