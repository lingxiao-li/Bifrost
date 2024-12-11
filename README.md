# BIFRÖST: 3D-Aware Image compositing with Language Instructions
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.19079-b31b1b)](https://arxiv.org/abs/2410.19079)&nbsp;
[![project page](https://img.shields.io/badge/Project%20page-SAR-pink)](https://lingxiao-li.github.io/bifrost.github.io/)&nbsp;

</div>

<p align="center">
  <img src="assets/visualization.png" width="720">
</p>

This repository includes the official pytorch implementation of BIFRÖST, presented in our paper:

**[BIFRÖST: 3D-Aware Image compositing with Language Instructions](https://arxiv.org/abs/)**

[Lingxiao Li](https://lingxiao-li.github.io/), [Kaixiong Gong](https://kxgong.github.io/), [Weihong Li](https://weihonglee.github.io/), [Xili Dai](https://delay-xili.github.io), [Tao Chen](https://eetchen.github.io/), [Xiaojun Yuan](https://yuan-xiaojun.github.io),[Xiangyu Yue](https://xyue.io/)

MMLab, CUHK & HKUST (GZ) & Fudan University & UESTC

Currently, we are working to organize the code.

## Update

- [2024.10] [arXiv](https://arxiv.org/abs/2410.19079) preprint is available.

## Introduction

This paper introduces **Bifröst**, a novel 3D-aware framework that is built upon diffusion models to perform instruction-based image composition. Previous methods concentrate on image compositing at the 2D level, which fall short in handling complex spatial relationships (*e.g.*, occlusion). *Bifröst* addresses these issues by training MLLM as a 2.5D location predictor and integrating depth maps as an extra condition during the generation process to bridge the gap between 2D and 3D, which enhances spatial comprehension and supports sophisticated spatial interactions. Our method begins by fine-tuning MLLM with a custom counterfactual dataset to predict 2.5D object locations in complex backgrounds from language instructions. Then, the image-compositing model is uniquely designed to process multiple types of input features, enabling it to perform high-fidelity image compositions that consider occlusion, depth blur, and image harmonization. Extensive qualitative and quantitative evaluations demonstrate that *Bifröst* significantly outperforms existing methods, providing a robust solution for generating realistically composited images in scenarios demanding intricate spatial understanding. This work not only pushes the boundaries of generative image compositing but also reduces reliance on expensive annotated datasets by effectively utilizing existing resources in innovative ways.

## Features

## Getting Started

### Prerequisites

We run the code on:

- Python 3.11
- PyTorch 2.3.1

## Acknowledgements

The code is built upon [ControlNet](https://github.com/lllyasviel/ControlNet), [AnyDoor](https://github.com/ali-vilab/AnyDoor). Thank for their great work.

## Citation

```bibtex
@INPROCEEDINGS{Li24,
  title = {BIFRÖST: 3D-Aware Image compositing with Language Instructions},
  author = {Lingxiao Li and Kaixiong Gong and Weihong Li and Xili Dai and Tao Chen and Xiaojun Yuan and Xiangyu Yue},
  booktitle={Advanced Neural Information Processing System (NeurIPS)},
  year={2024}
}
```
You can contact me via email lxili@ie.cuhk.edu.hk if any questions.
