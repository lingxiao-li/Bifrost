# BIFRÖST: 3D-Aware Image compositing with Language Instructions
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.19079-b31b1b)](https://arxiv.org/abs/2410.19079)&nbsp;
[![project page](https://img.shields.io/badge/Project%20page-Bifrost-pink)](https://lingxiao-li.github.io/bifrost.github.io/)&nbsp;

</div>

<p align="center">
  <img src="assets/Figure_1.png" width="720">
</p>

This repository includes the official pytorch implementation of BIFRÖST, presented in our paper:

**[BIFRÖST: 3D-Aware Image compositing with Language Instructions](https://arxiv.org/abs/2410.19079)**

[Lingxiao Li](https://lingxiao-li.github.io/), [Kaixiong Gong](https://kxgong.github.io/), [Weihong Li](https://weihonglee.github.io/), [Xili Dai](https://delay-xili.github.io), [Tao Chen](https://eetchen.github.io/), [Xiaojun Yuan](https://yuan-xiaojun.github.io),[Xiangyu Yue](https://xyue.io/)

MMLab, CUHK & HKUST (GZ) & Fudan University & UESTC

Currently, we are working to organize the code.

## Update

- [2025.1] Code and weights are released!
- [2024.10] [arXiv](https://arxiv.org/abs/2410.19079) preprint is available.

## Introduction

This paper introduces **Bifröst**, a novel 3D-aware framework that is built upon diffusion models to perform instruction-based image composition. Previous methods concentrate on image compositing at the 2D level, which fall short in handling complex spatial relationships (*e.g.*, occlusion). *Bifröst* addresses these issues by training MLLM as a 2.5D location predictor and integrating depth maps as an extra condition during the generation process to bridge the gap between 2D and 3D, which enhances spatial comprehension and supports sophisticated spatial interactions. Our method begins by fine-tuning MLLM with a custom counterfactual dataset to predict 2.5D object locations in complex backgrounds from language instructions. Then, the image-compositing model is uniquely designed to process multiple types of input features, enabling it to perform high-fidelity image compositions that consider occlusion, depth blur, and image harmonization. Extensive qualitative and quantitative evaluations demonstrate that *Bifröst* significantly outperforms existing methods, providing a robust solution for generating realistically composited images in scenarios demanding intricate spatial understanding. This work not only pushes the boundaries of generative image compositing but also reduces reliance on expensive annotated datasets by effectively utilizing existing resources in innovative ways.

## Installation

### Installation with MLLM
Follow the instruction with the original LLaVA installation, check LLaVA_Bifrost/README.md

### Installation with Image Compositing
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate bifrost
```
or `pip`:
```bash
pip install -r requirements.txt
```
Additionally, for training, you need to install panopticapi, pycocotools, and lvis-api.
```bash
pip install git+https://github.com/cocodataset/panopticapi.git

pip install pycocotools

pip install lvis
```
## Download Checkpoints
Download Bifrost checkpoint: 
* [BaiduNetDisk](https://pan.baidu.com/s/1_nHmskl3z5TfcuwEzBRVBA?pwd=czhc)

Download DINOv2 checkpoint and revise `/configs/anydoor.yaml` for the path (line 103)
* URL: https://github.com/facebookresearch/dinov2?tab=readme-ov-file

Download Stable Diffusion V2.1 if you want to train from scratch.
* URL: https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main

We also support train from the checkpoint of [AnyDoor](https://github.com/ali-vilab/AnyDoor)
* URL: https://modelscope.cn/models/damo/AnyDoor/files


## Inference
### Inference Bifrost Image Compositing
The code is provided in the folder Main_Bifrost
We provide inference code in `run_inference.py` (from Line 370 - ) for both inference of a single image and inference of a dataset (DreamBooth Test). You should modify the data path and run the following code. 
The generated results are provided in the path you set.

```bash
sh scripts/inference.sh
```

### Inference Bifrost MLLM
The code is provided in the folder LLaVA_Bifrost
First place the downloaded checkpoint folder in LLaVA_Bifrost/llava/checkpoints
We provide inference code in `LLaVA_Bifrost/llava/eva/run_llava.py` (from Line 154 - ). You should modify the data path and run the following code. 
The generated results are provided in the path you set.

```bash
python llava/eva/run_llava.py
```

## Train

### Train MLLM

#### Prepare datasets for fine-tuning MLLM
* Download [MS-COCO](https://cocodataset.org/) dataset
* Download the corresponding code and weight of [DPT](https://github.com/isl-org/DPT)(to predict the depth) and [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything)(to fill the hole) firstly.
* Using the script we provided in LLaVA_Bifrost/create_dataset.py to create the customized counterfactual dataset. You should modify the data path and run the following code.
  
```bash
python create_dataset.py
```

#### Prepare initial weight
* Download the initial weight from [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b).
* You should modify the data path.

#### Start training
* Modify the training hyper-parameters in `train.sh`.
* You should modify the data path.

* Start training by executing: 
```bash
sh train.sh  
```

### Train Image Compositing model

#### Prepare datasets for Image Compositing
* Download the datasets that are present in `/configs/datasets.yaml` and modify the corresponding paths.
* You could prepare your own datasets according to the format of files in `./datasets`.
* If you use Uthe VO dataset, you need to process the json following `./datasets/Preprocess/uvo_process.py`
* You could refer to `run_dataset_debug.py` to verify your data is correct.


#### Prepare initial weight
* If you would like to train from scratch, convert the downloaded SD weights to control copy by running:
```bash
sh ./scripts/convert_weight.sh  
```

* If you would like to train from the checkpoint of Anydoor, convert the download AnyDoor checkpoint to control copy by running:
```bash
sh ./scripts/convert_weight_anydoor.sh 
```

#### Start training
* Modify the training hyper-parameters in `run_train_bifrost.py` Line 29-38 according to your training resources. We verify that using 2-A100 GPUs with batch accumulation=1 could get satisfactory results after 200,000 iterations. (You need at least one A100 GPU to train the model.)


* Start training by executing: 
```bash
sh ./scripts/train.sh  
```

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
