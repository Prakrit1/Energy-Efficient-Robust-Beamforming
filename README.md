# Energy-Efficient Robust Beamforming



The code version associated with this paper is found in the releases. The project structure is as follows

```
.
├── models                  | trained models (download from Releases)
├── outputs
│   └── metrics             | metrics from training/evaluation
├── README.md               | this file
├── requirements.txt        | project dependencies
├── environment.yml         | conda environment (GPU)
├── LICENSE                 | license
├── src                     | python source related to..
│   ├── config              |   configuration
│   ├── data                |   data generation, i.e., satellite model
│   ├── models              |   learning models
│   ├── energy_efficiency   |   evaluation and figure scripts
│   ├── plotting            |   plotting
└── └── utils               |   shared helper functions
```

## Install

GPU (recommended):
```bash
conda env create -f environment.yml
conda activate gpu_beamforming
```
CPU / pip:
```bash
pip install -r requirements.txt
```
Tested with TensorFlow 2.15 on an NVIDIA RTX 8000.

## Running

Put the repo root on `PYTHONPATH` first (every script imports `src.*`):
```bash
export PYTHONPATH=$(pwd)
```

Train the energy-efficiency (EE) agent &mdash; the main entry point is `src/models/EE_sac.py`:
```bash
EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive \
EE_TRAIN_ERROR_BOUND=0.0 \
EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000 \
python src/models/EE_sac.py
```
- `EE_TRAIN_ERROR_BOUND` &mdash; CSIT AoD error bound to train under (e.g. `0.0`, `0.025`, `0.05`).
- `EE_DINKELBACH_LAMBDA_WINDOW_STEPS` &mdash; EMA window for the Dinkelbach lambda.

Train the rate-only baseline (RM) for comparison:
```bash
EE_REWARD_MODE=sum_rate_only EE_TRAIN_ERROR_BOUND=0.0 python src/models/EE_sac.py
```

Checkpoints are written to `models/<training_name>/`; metrics to
`outputs/metrics/<training_name>/`. To reproduce the figures without retraining,
download the learned models from the Releases into `models/`, then run the
evaluation scripts:
```bash
python src/energy_efficiency/plotting_scenario.py             # sum-rate vs. CSIT-error sweep
python src/energy_efficiency/error_sweep_training_triplet.py  # EE across three error bounds
python src/energy_efficiency/ee_power_rate_tradeoff_sac.py    # rate-vs-power trade-off
```

## License
See [LICENSE](LICENSE).
