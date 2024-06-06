# MMDetection-DS


[中文](README-zh.md) | [English](README-zh)

A foundational codebase for **dual-spectrum image** object detection, based on the [MMDetection3.1.0](README-en.md) framework code.

**Related**

- [MMDetection v3.1.0 Official Documentation](https://mmdetection.readthedocs.io/zh-cn/v3.1.0/)
- [MMDetection v3.1.0 Official Codebase](https://github.com/open-mmlab/mmdetection/tree/v3.1.0)

**Main Features**

- Customized aligned dual-spectrum dataset configuration files
- Customized dual-spectrum dataset preprocessing configuration
- Customized dual-stream network structure

**Background**

Recent dual-spectrum object detection works claim to have achieved high accuracy on multiple dual-spectrum datasets, but none have open-sourced their code, making their methods difficult to reproduce and learn from.

Additionally, current object detection works heavily utilize various optimization tools, making it challenging to compete with SOTA methods that incorporate numerous optimization tools. Frameworks like [Detection2](https://github.com/facebookresearch/detectron2) and [MMDetection](https://github.com/open-mmlab/mmdetection) are highly extensible and user-friendly, allowing easy use of various optimization tools, but their extensive abstraction and encapsulation make them challenging for beginners to learn.

Therefore, we hope to open-source a dual-spectrum object detection codebase to facilitate better learning and research on dual-spectrum object detection tasks.

## 1. Environment Setup

Please follow the steps below to configure the runtime environment:

### 1. First, clone this codebase

```shell
git clone --depth 1 <this repo url> mmdetection-ds
cd mmdetection-ds
```

### 2. Create an environment and install dependencies

Create and activate a new conda environment

```shell
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

Install PyTorch, the default installation is the CPU version. For the GPU version, please refer to 1. [CUDA-Toolkit version 11.8 or above](https://pytorch.org/get-started/locally/), 2. [Other lower versions](https://pytorch.org/get-started/previous-versions/)

```shell
conda install pytorch torchvision -c pytorch  
# It is recommended not to use this command directly, but to choose the appropriate version based on the links above
```

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"
pip install -v -e .
```

## 2. Dataset Preparation

Aligned dual-spectrum (RGB+IR) datasets

### LLVIP Dataset

**LLVIP Dataset**: https://bupt-ai-cz.github.io/LLVIP/

**Note**: If you do not want to use the following script to convert annotation files or find it inconvenient to use the LLVIP provided dataset download method, you can directly use our backup (including converted annotations), download link: https://huggingface.co/datasets/UserNae3/LLVIP

**Dataset Download and Annotation Conversion:**

The annotation files provided by LLVIP are in VOC format. Although they provide a format conversion script, all annotations are in one folder without separation.

We provide a script that uses their conversion script to automatically generate corresponding annotation files based on the training and test set splits of the dataset.

First, download the LLVIP dataset zip file `LLVIP.zip` and extract it to a folder, then run the following script

```shell
# We use the typer library, so you need to install it first
pip install typer

cd dataset/ # Enter the dataset/ folder of this project
python dct.py llvip --data_dir path_to_extracted_LLVIP_dataset
```

It will generate COCO format annotation files for the training and test sets and save them in the `coco_annotations` folder.

### FLIR Dataset

**FLIR Dataset**: https://www.flir.com/oem/adas/adas-dataset-form/

The FLIR dataset has multiple versions. The aligned dual-spectrum dataset generally uses the version from Zhang et al. [Paper](https://arxiv.org/abs/2009.12664), [Dataset](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view), hereafter referred to as the `FLIR dataset`.

**Note**: If you do not want to use the following script to convert or find it inconvenient to use the dataset download method provided by Zhang et al., you can directly use our backup (including converted annotations), download link: https://huggingface.co/datasets/UserNae3/FLIR_aligned

**Dataset Conversion:**

Similar to the LLVIP dataset, the FLIR dataset also provides VOC format annotation files. We provide a script that calls their conversion script to automatically generate corresponding annotation files based on the training and test set splits of the dataset.
First, download the FLIR dataset zip file `FLIR.zip` and extract it to a folder, then run the following script

```shell
cd dataset/ # Enter the dataset/ folder of this project
python dct.py flir --data_dir path_to_extracted_FLIR_dataset
```

Additionally, in some past papers, the dog category is generally removed from the FLIR dataset due to its small quantity. We also provide a script to remove specific categories from COCO annotations. This is also included in the backup we provided, which includes annotations with the dog category removed.

```shell
cd dataset/ # Enter the dataset/ folder of this project
python python dct.py coco-rc --help # View usage help
```

## 3. Related Configuration Files and Models

**Training Configuration Files:** [aligned_dual_spectrum_od](configs/aligned_dual_spectrum_od)

**Dual-Spectrum Dataset Configuration Files:** [aligned_dual_spectrum_dataset.py](mmdet/datasets/aligned_dual_spectrum_dataset.py)

**Dual-Spectrum Dataset Preprocessing Configuration Files:** [aligned_dual_spectrum_preprocess.py](mmdet/models/data_preprocessors/data_preprocessor.py)

**Example Model:** [ts_backbone.py](mmdet/models/backbones/ts_backbone.py)

## 4. More

For more resources, refer to

- Multispectral-Pedestrian-Detection-Resource: https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource

## 5. Copyright

> The copyright of the models, datasets, and optimization tools involved in this code belongs to the original authors. Please refer to the relevant literature to understand the related copyright information.

 

