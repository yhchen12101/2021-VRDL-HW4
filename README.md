# 2021-VRDL-HW4

This repository contains the code for homework 4 of 2021 Fall Selected Topics in Visual Recognition using Deep Learning.

## Requirements
```
Pillow
torch==1.7.0
torchvision==0.8.1
PyYAML>=5.3.1
tqdm>=4.41.0
imageio
tensorboardX
six
```

## Dataset Preparation
1. Download the dataset from the [Google Drive](https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view)
2. Create a folder `mkdir dataset` to put the dataset 
3. Unzip the downloaded dataset in `dataset` folder. The file structire will be as following:
```
dataset/
├── training_hr_images/
    ├── 2092.png
    ├── 8049.png
    │     .
    │     .
    │     .
    ├──
├── testing_lr_images/
    ├── 00.png
    ├── 01.png
    │     .
    │     .
    ├── 13.png
```

## Quick Start for generating the submitted results
1. Create a folder `mkdir pictures` to put the generated images 
2. Download the [model weight](https://drive.google.com/file/d/1zrVt7B_NM12dVmkWVcg2JskBaUsZ0-9w/view?usp=sharing) and run
```
python demo.py --output ./pictures/ --model epoch-last.pth
```

## Train a Model
Run the comment to train EDSR model, where the config can be repalced by other model's config file when training other model.
```
python train.py --config ./configs/train_edsr.yaml --name edsr
```

## Acknowledgement
This implementation is heavily based on [LIIF](https://github.com/yinboc/liif).
