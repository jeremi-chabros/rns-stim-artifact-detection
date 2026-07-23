# RNS Stimulation Artifact Detection

Waveform-only detection and masking of responsive neurostimulation (RNS)
stimulation artifacts in intracranial EEG (ECoG), using a 1-D U-Net. The model
operates on the raw multi-channel 250 Hz waveform alone — no device metadata at
inference — and produces sample-level artifact masks that transfer across
epilepsy syndromes without retraining.

This repository contains the source code and evaluation harnesses accompanying
the manuscript *"Waveform-Only Detection of RNS Stimulation Artifacts that
Transfers Across Epilepsy Syndromes"* (under review).

## Repository layout

- `src/` — catalog building, mask refinement, the U-Net model, training/eval
  drivers, model-free baselines, and metrics
- `src/figures/` — manuscript figure generators
- `tests/` — unit tests
- `main.py` — entry point

## Requirements

- Python ≥ 3.14, managed with [uv](https://docs.astral.sh/uv/)
- PyTorch 2.x (MPS or CUDA)

```bash
uv sync
```

## Data

The intracranial recordings and device-logged therapy metadata used in the paper
are governed by institutional data-use agreements and IRB protocols and are
**not** included in this repository. Access is available to qualified
investigators as described in the manuscript's Data Availability statement.

## Model weights

Trained model weights are distributed separately as a release asset (see the
Releases tab) rather than tracked in the repository.

## License

Released under the MIT License. See [LICENSE](LICENSE).
