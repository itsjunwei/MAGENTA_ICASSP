# MAGENTA: Magnitude and Geometry-Enhanced Training Approach for Robust Long-Tailed Sound Event Localization and Detection (SELD)

[![arXiv](https://img.shields.io/badge/arXiv-2509.15599-b31b1b.svg)](https://arxiv.org/abs/2509.15599)
[![License: TBD](https://img.shields.io/badge/License-TBD-lightgrey.svg)](TBD)

This repository contains the official PyTorch implementation for the paper: **MAGENTA: Magnitude and Geometry-Enhanced Training Approach for Robust Long-Tailed Sound Event Localization and Detection**.

Any questions should be directed to ``junwei004@e.ntu.edu.sg``

## Description

**MAGENTA** is a unified loss function designed to address the challenge of severe class imbalance in real-world, long-tailed datasets for Sound Event Localization and Detection (SELD). Standard regression losses like MSE often cause models to under-recognize rare events. MAGENTA tackles this by geometrically decomposing the regression error into radial (activity) and angular (direction) components within the Activity-Coupled Cartesian DOA (ACCDOA) vector space. This allows for targeted, rarity-aware penalties that combat detection timidity and improve directional accuracy for infrequent sound classes, without requiring synthetic data.

This implementation provides a drop-in replacement for standard SELD loss functions and includes code for all ablations discussed in the paper.


## Usage

The core loss function is implemented in `magenta_loss.py`. It can be easily integrated into any SELD training pipeline that uses the multi-ACCDOA output format.

A factory function `build_magenta_ablation` is provided to instantiate the loss function corresponding to the experiments in the paper.

## Citation

If you find our code useful in your work, please do cite our paper:

```
@article{yeow2025magenta,
  title={MAGENTA: Magnitude and Geometry-ENhanced Training Approach for Robust Long-Tailed Sound Event Localization and Detection},
  author={Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng},
  journal={arXiv preprint arXiv:2509.15599},
  year={2025}
}
```
