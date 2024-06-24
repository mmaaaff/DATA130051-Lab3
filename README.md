<h1 align="center">DATA130051 Project2</h1>

<div align="center">周语诠</div>
<div align="center">2024-5-31</div>

## Contents

- [Contents](#contents)
- [Description](#description)
- [Preparation](#preparation)
- [Training](#training)
- [Testing](#testing)


## Description

This lab trains trains ViT and ResNet models on CIFAR-100 dataset to compare these two types of models. 

In the root directory:

1. **myvit.py** -- my implementation of ViT using PyTorch
2. **resnet.py** -- an implementation of ResNet using PyTorch
3. **ViT_train.ipynb** -- training script for ViT
4. **ResNet_train.ipynb** -- training script for ResNet

***

## Preparation

1. **Environment**
    Basic PyTorch environment is required. 

2. **Preparation**
    Download dataset and trained models from https://1drv.ms/f/s!Avf2C4ss_613kOgN0XdmClFwxS9M2Q?e=y6uQp0. Place folder **data** and **weights** at root directory. You can choose not to download them as dataset will be downloaded when running the training script if not found.

***

## Training

    To train the model from scratch, run the training script in Jupyter Notebook directly. For ViT, you can change model structure in the "Define the Model". A typical setting is **model = VisionTransformer(img_size=32, patch_size=4, d_model=48\*16, num_heads=16, mlp_dim=48*8*3, num_layers=6, num_classes=100)**. You need to make sure that d_model is divisible by num_heads, otherwise it will raise an error because multiheads will not be successfully split. For ResNet, you can use different models from **resnet.py** in the "Define the Model" part.

***

## Testing

    Testing scripts is included in training scripts. Training and validation loss and validation accuracy will be printed while training, and they will be written to the tensorboard file. You can use tensorboard to visualize the training process.