This code was used in the following work.

[1] Prakrit Parajuli, Alea Schröder, Dirk Wübben, Armin Dekorsy, "Energy-Efficient Robust Beamforming for LEO
Satellite Downlink via Reinforcement Learning".

Email: {parajuli, schroeder, wuebben, dekorsy}@ant.uni-bremen.de


The code version associated with this paper is found in the releases. The project structure is as follows. Need to rerun the code to generate trained models and output figures.

```
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


## License
See [LICENSE](LICENSE).
