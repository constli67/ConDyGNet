# ConDyGNet

Official PyTorch implementation of **ConDyGNet: Constraint-Guided Dynamic Graph Networks for Multivariate Time Series Forecasting**, accepted by **IJCAI-ECAI 2026**.

## Overview

ConDyGNet models robust time-varying inter-channel dependencies with the principle of **global basis, dynamic weights**. It learns a low-rank global basis as a shared structural constraint and generates patch-wise mixing weights to construct dynamic propagation graphs, balancing structural stability and temporal adaptivity.

## Requirements

The main dependencies are:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Data

Place the eight benchmark datasets under `./dataset/` as follows:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/electricity.csv
├── weather/weather.csv
├── solar-energy/solar_AL.txt
└── traffic/traffic.csv
```

## Training and Evaluation

Run the corresponding script to train and evaluate ConDyGNet. For example:

```bash
bash ./scripts/ETT_script/ConDyGNet_ETTh1.sh
```

Scripts for all datasets and forecasting horizons are provided in `./scripts/`. Model checkpoints are saved to `./checkpoints/`, and evaluation results are written to `result_long_term_forecast.txt`.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{li2026condygnet,
  title     = {ConDyGNet: Constraint-Guided Dynamic Graph Networks for Multivariate Time Series Forecasting},
  author    = {Li, Zhenzhou and Li, Xiang and Niu, Zhibin},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year      = {2026}
}
```

## Acknowledgements

This repository is developed based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library). We thank its authors for providing the benchmark framework.
