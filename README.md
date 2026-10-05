# Energy-Efficient Robust Beamforming

This code was used in the following unpublished work (preprint not available yet).

[1] Alea Schröder, Steffen Gracla, Carsten Bockelmann, Dirk Wübben, Armin Dekorsy, "Model-free Robust Beamforming in Satellite Downlink using Reinforcement Learning", under review.

Email: {schroeder, gracla, bockelmann, wuebben, dekorsy}@ant.uni-bremen.de

The code version associated with this paper along with the used learned models and evaluation results is found in the releases. The project structure is as follows

```
.
├── models                  | trained models (download from Releases)
├── outputs
│   └── metrics             | metrics from training / evaluation
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

Every script imports `src.*`, so put the repo root on `PYTHONPATH` first, then
run the main entry point:
```bash
export PYTHONPATH=$(pwd)
python src/models/EE_sac.py
```
Run behaviour (reward mode, CSIT error bound, etc.) is set through environment
variables read at the top of `src/models/EE_sac.py`.

## Trained models

Training saves checkpoints inside the cloned repo, under:
```
models/<training_name>/base/full_snap_energy_effiency_<EE>/
├── model/weights.weights.h5   | network weights
└── config/                    | config + normalization needed to reload the model
```
`<training_name>` encodes the run settings (e.g. `EE_dinkelbach_adaptive`). Only
the best-scoring checkpoints are kept, so the folder with the highest `<EE>` in
its name is the best model. Training metrics are written to
`outputs/metrics/<training_name>/base/`.

To use the paper's models without retraining, download them from the Releases
and extract into `models/` (so the paths above exist), then run the figure
scripts in `src/energy_efficiency/`.

## License
See [LICENSE](LICENSE).
