# MMDetection-DS

用于**双光谱图像**的目标检测的基础代码库，它基于[MMDetection3.1.0](README-en.md)框架代码。

**相关**

- [MMDetectionv3.1.0官方文档](https://mmdetection.readthedocs.io/zh-cn/v3.1.0/)
- [MMDetectionv3.1.0官方代码库](https://github.com/open-mmlab/mmdetection/tree/v3.1.0)


**主要特性**

- 定制的对齐的双光谱数据集，配置文件
- 定制的双光谱数据集预处理配置
- 定制的双流网络结构

**缘起**

最近的一些双光谱目标检测工作声明自己在多个双光谱数据集上达到了很好的精度，但是它们都没有开放它们的源代码，使得方法难以复现和学习。

另外现在目标检测工作大量使用了诸多优化工具，附加诸多优化工具的SOTA挑战起来变得困难， [Detection2](https://github.com/facebookresearch/detectron2)
和 [MMDetection](https://github.com/open-mmlab/mmdetection)
两个深度学习目标检测框架，虽然都有着高度可扩展性易用性并且可以轻松使用诸多优化工具，但它们做了大量的抽象和封装，对于新手来说学习成本较高。

因此，我们希望能够开放一个双光谱目标检测的代码库，以便于研究者们能够更好的学习和研究双光谱目标检测任务。

## 安装

请按照以下步骤配置运行环境:

### 1. 首先clone 本代码库

```shell
git clone  --depth 1 <this repo url> mmdetection-ds
cd mmdetection-ds
```

### 2. 创建环境和安装依赖

创建 并激活一个新的conda环境

```shell
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

安装pytorch,默认安装cpu版本, 请安装gpu版本，请查看 1. [CUDA-Toolkit版本为11.8以上](https://pytorch.org/get-started/locally/),2. [其他低版本](https://pytorch.org/get-started/previous-versions/)

```shell
conda install pytorch torchvision -c pytorch  
# 建议不要直接使用这个命令，而是根据上面查看的链接选择合适的版本
```

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"
pip install -v -e .
```

## 数据集准备

对齐的双光谱（RGB+IR）数据集

### LLVIP 数据集

**LLVIP 数据集** ： https://bupt-ai-cz.github.io/LLVIP/

**注意**：如果你不想使用以下的脚本转换标注文件或不方便使用LLVIP 提供的数据集下载方式，你都可以直接使用我们备份(
包含转换好的注释)，下载地址为：https://huggingface.co/datasets/UserNae3/LLVIP

**数据集下载和注释转换：**

在LLVIP 提供的标注的标注文件为VOC格式，虽然它们提供了格式转换脚本，但是所有的标注都在一个文件夹中没有区分开。

我们提供了一个脚本利用它们提供的转换脚本，自动根据数据集训练集和测试集的划分生成对应的标注文件。

首先下载LLVIP数据集压缩包`LLVIP.zip`，并将其解压到一个文件夹中, 然后运行以下脚本

```shell
# 我们使用了 typer 这个库，所以你需要首先安装它
pip install typer

cd dataset/ # 进入本项目 dataset/ 文件夹
python dct.py llvip --data_dir 你解压的LLVIP数据集路径
```

它将会生成coco格式的训练集和测试机的标注文件，并保存在`coco_annotations`文件夹中。

### FLIR 数据集

**FLIR数据集**：https://www.flir.com/oem/adas/adas-dataset-form/

FLIR 数据集有多个版本，其中对齐的双光谱数据集一般是使用来自
Zhang等人的版本[Paper](https://arxiv.org/abs/2009.12664), [Dataset](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view)
，以下称为`FLIR数据集`。

**注意：** 如果你不想使用以下的脚本转换或不方便使用Zhang 等人提供的数据集下载方式，你都可以直接使用我们备份(
包含转换好的注释)，下载地址为：https://huggingface.co/datasets/UserNae3/FLIR_aligned

**数据集转换：**

和LLVIP数据集一样，FLIR数据集也提供了VOC格式的标注文件，我们提供了脚本调用它们提供的转换脚本，自动根据数据集训练集和测试集的划分生成对应的标注文件。
首先下载FLIR数据集压缩包`FLIR.zip`，并将其解压到一个文件夹中, 然后运行以下脚本

```shell
cd dataset/ # 进入本项目 dataset/ 文件夹
python dct.py flir --data_dir 你解压的FLIR数据集路径
```

另外，在过往一些paper中，对于FLIR 数据集，一般会移除其中的 dog 类别，因为它的数量太少，我们也提供了的脚本可以用来移除COCO标注中移除特定类别
当然这在上面我们提供的备份中也包括了移除dog类别的注释

```shell
cd dataset/ # 进入本项目 dataset/ 文件夹
python python dct.py coco-rc --help # 查看使用帮助
```

## 模型

...

### 更多

更多资源可参考

- Multispectral-Pedestrian-Detection-Resource：https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource
- multispectral-object-detection: https://github.com/DocF/multispectral-object-detection