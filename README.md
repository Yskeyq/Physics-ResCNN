# Physics-ResCNN

Software for **Surrogate-Assisted Physics-Guided Retrieval of Range-Varying
Evaporation Duct Height From Radar Sea Clutter** (Yi Yan et al.).

This is release 1.2.0 associated with the first manuscript revision.
The manuscript is under review; no acceptance or publication DOI is implied.
The repository URL and separate dataset accession have not yet been assigned.

## What the model does

The deployed inverse network maps one profile-centered 500-point synthetic clutter
sequence, in dB, to 101 EDH samples and duct-presence indicators over 0--100 km.
The 1-km interval is output sampling, not an experimentally established independent
spatial resolution. Only clutter is supplied online. Radar/scattering terms,
meteorology and the frozen forward surrogate are used offline or during training.

The benchmark uses synthetic observations from the East China Sea in 2025 under
one radar configuration. The release is not a field-measurement dataset.

## Installation

Use Python 3.10 or newer in an isolated environment. Install PyTorch for the
appropriate CPU/CUDA platform, then install `requirements.txt`. Exact versions of
the tested environment are recorded in `requirements-tested.txt`; that file does
not select a CUDA wheel index. No dependency source is bundled.

## Separate data and model files

The planned data deposit is an **IEEE DataPort Standard Dataset**. Its DOI is
pending. Standard-file access may require an IEEE DataPort subscription; do not
interpret this code repository as providing unrestricted access to those files.

The root MIT license covers the software, not the separately supplied data or
checkpoints. Own data/weight contributions use CC BY 3.0 US, with underlying
third-party terms retained. They remain outside the GitHub code archive.

After obtaining the deposit, place its contents inside `data/`, or set
`PHYSICS_RESCNN_DATA` to the extracted data directory. Expected structure:

```
data/
  pc_dataset_v1/          # inverse train / validation / frozen joint test
  pe_reference/          # cached PE references for all five development roles
  model_role_split/      # frozen day-grouped manifests and indices
  models/                # selected inverse models and frozen forward model
  records/               # frozen sensitivity, transfer, mismatch and other results
  geometry/              # frozen plotting geometry
  examples/              # first eight validation clutter profiles, no targets
```

Model exports preserve every state-dictionary tensor exactly. Optimizer state and
machine-specific metadata are removed. Therefore serialized checkpoint SHA-256
values differ from historical files; the release manifest records new hashes.
Only load trusted checkpoints and verify the manifest. The prediction entrypoint
uses PyTorch's `weights_only=True` reader.

## Clutter-only inference

Input is an NPZ with `pc_centered_dB` of shape `(N, 500)`, float32, finite, with
each profile mean subtracted over the same 500 range points. This entrypoint does
not load EDH targets, the forward model, PE outputs, or the radar term.

```
python src/predict.py --input data/examples/clutter_validation_first8.npz --output artifacts/predictions.npz
```

Outputs are `edh_m`, `duct_mask`, `duct_probability` and `range_km`. Heights are
the nonnegative EDH head multiplied by the predicted binary mask at probability
threshold 0.5. Outputs are not overwritten. Use `--device cuda` explicitly for
GPU inference. The fixed-radar synthetic-input control is intentionally excluded
from the ordinary clutter-only prediction interface.

## Training the principal comparison

Only `inverse_train` and `inverse_validation` are opened. Statistics, sampling and
checkpoint choice remain training/development-only. New training outputs go to
`artifacts/`, never to the frozen deposited `models/` directory.

```
python src/train_inverse_baseline_v2.py --physics-loss-weight 0 --output-dir artifacts/supervised
python src/train_inverse_baseline_v2.py --physics-loss-weight 0.01 --physics-scale-db 5 --output-dir artifacts/physics
```

Both use seed 20260813 by default. The physics run requires the frozen forward
checkpoint at `data/models/forward_rescnn/best_model.pt`. No PE call occurs in
inverse training. To train the forward surrogate separately from cached PE:

```
python src/train_forward_rescnn.py data/pe_reference/pe_ls_dataset_zs_0p3m_v2.mat data/model_role_split
```

The nominal setting is a fixed design choice supported by a subsequent local
validation study, not a claim of global hyperparameter optimality. Its complete
six-setting, three-seed records are in `data/records/sensitivity/`.

## Checks before a long run

```
python -m unittest discover -s tests -v
python src/train_forward_rescnn.py --self-test
python src/train_inverse_baseline_v2.py --self-test
python src/evaluate_joint_test.py --dry-run
python src/plot_manuscript_figures_v6.py --dry-run
```

The dry-run evaluator checks all seven formal checkpoint selections without
opening joint-test data. The adapted RFCFormer selection is the low-dropout v2
run, not an obsolete result. Read `NOTICE.md` for its attribution and task scope.

A short physical-training pipeline test can be run explicitly:

```
python src/train_inverse_baseline_v2.py --smoke-test --smoke-epochs 1 --smoke-train-samples 64 --smoke-validation-samples 32 --physics-loss-weight 0.01 --output-dir artifacts/smoke_physics
```

This is a software smoke test, not a new benchmark result. It must not replace a
frozen manuscript checkpoint.

## Evaluation and plotting

`evaluate_joint_test.py` requires the explicit `--confirm-final-test` option before
opening the frozen test cohort. Running it does not authorize choosing new seeds,
settings or checkpoints from test results. Existing original per-sample results
are supplied under `records/joint_test_final_rfcformer_v2/`.

`plot_manuscript_figures_v6.py` reads frozen geometry, robustness and ambiguity
records only; it performs no training or inference. It regenerates benchmark
plots, not manually composed schematic elements. The dedicated broken-axis
robustness plot is in `plot_robustness_broken_axis_v6_1.py`.

Cached mismatch, directional-response and direct-PE posterior outputs permit
inspection/reaggregation of the reported evidence. All 18 sensitivity checkpoints,
regional forward/inverse checkpoints and regional development inputs are included
in the separate data archive, along with three loss ablations and the distinct
fixed-reference-in-physics-loss control. The private PE solver source is not
distributed, so this release does not reproduce new PE calculations from scratch.
See `data/DATA_DICTIONARY.md` for role/index conventions and artifact coverage.

```
python src/audit_cached_results.py
python src/reproduce_training.py --suite sensitivity
python src/reproduce_training.py --suite spatial
python src/reproduce_training.py --suite fixed-reference
```

The audit reaggregates cached outputs without new test inference. Training-plan
commands only print commands by default; `--execute` starts full runs in new output
directories. Spatial plans train a separate forward model per direction/seed and
use that matching model in the regional inverse run. Global and regional core
model/loss definitions match the frozen original implementations. Retraining on
different hardware need not yield bitwise-identical weights or scores. The frozen
checkpoints remain authoritative for manuscript values.

The three other loss ablations have their own launcher:
`python src/train_inverse_ablation_suite.py --dry-run`.
Fixed-radar input construction and the fixed radar reference INSIDE the physics
loss are different controls; see the data dictionary. The latter trainer is
`train_inverse_physics_radar_ablation_v0_7.py`.

## License and citation

Original software: MIT, see `LICENSE`. Data, checkpoints and third-party material:
separate terms, see `NOTICE.md` and the deposit. Please cite the manuscript and
the dataset when its DOI is available. `CITATION.cff` identifies the work without
inventing an article DOI or claiming acceptance.
