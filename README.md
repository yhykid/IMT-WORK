
<div align="center" markdown>

# 图像测量技术实验作业


</div>

## Introduction

Use mindsopre framework and mindcv library to complete the implementation of Swin Transformer algorithm.Swin-Transformer is the best paper of ICCV 2021.
Swin Transformer is a deep learning model based on Transformer with SOTA performance in the visual field. It has better performance and accuracy than ViT.

The version of  **MindSpore is 2.1.1**.

## Environment Creation

- **CUDA version is 11.6:.** 

    ```pycon
    # Download Code
    >>> git clone https://github.com/ChaselLau666/Image-Measurement-Technology-Work.git
    # create a conda environment
    >>> conda create -n ms python=3.8
    # Install mindspore 2.1.1
    >>> conda install mindspore=2.1.1 -c mindspore -c conda-forge
    #install rely
    >>> pip install tqdm
    >>> pip install pyyaml
    ```   
    See [Installation](https://www.mindspore.cn/install/) for details


## Data set download

You should Install download library

```pycon
pip install download
```

Below is a code  to download the dataset

```pycon
from download import download
dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/vit_imagenet_dataset.zip"
path = "./"
path = download(dataset_url, path, kind="zip", replace=True)
```



**Loss Curve**


<p align="center">
  <img src="https://github.com/ChaselLau666/Image-Measurement-Technology-Work/blob/main/img/loss.png?raw=true
" width=360 />
</p>


## Training

It is easy to train  model on a standard or customized dataset using `train.py`.
The mindcv module provides relevant definition components. You only need to define the location of the data set and the relative parameters of the model. The model parameters have been encapsulated in `ckg`.

- Model Parameters

    ```shell
    mode: 0
    distribute: True
    num_parallel_workers: 1
    val_while_train: True
    val_interval: 1
    # dataset
    dataset: "imagenet"
    data_dir: "/path/to/imagenet"
    shuffle: True
    dataset_download: False
    batch_size: 16
    drop_remainder: True
    # augmentation
    image_resize: 224
    scale: [0.08, 1.0]
    ratio: [0.75, 1.333]
    hflip: 0.5
    interpolation: "bicubic"
    re_prob: 0.1
    mixup: 0.2
    cutmix: 1.0
    cutmix_prob: 1.0
    crop_pct: 0.875
    color_jitter: [0.4, 0.4, 0.4]
    auto_augment: "randaug-m7-mstd0.5"

    # model
    model: "swin_tiny"
    num_classes: 1000
    pretrained: False
    ckpt_path: ""
    keep_checkpoint_max: 1
    ckpt_save_policy: "top_k"
    ckpt_save_dir: "./ckpt"
    epoch_size: 1000
    dataset_sink_mode: True
    amp_level: "O2"

    # loss
    loss: "CE"
    loss_scale: 1024.0
    label_smoothing: 0.1

    # lr scheduler
    scheduler: "cosine_decay"
    lr: 0.00006
    min_lr: 1e-6
    warmup_epochs: 32
    decay_epochs: 568
    lr_epoch_stair: False

    # optimizer
    opt: "adamw"
    weight_decay: 0.025
    filter_bias_and_bn: True
    use_nesterov: False
    ```

    Above are the model parameters
- Training

    ```shell
    # training
    python train.py --config /home/xiangchengliu/Videos/Image-Measurement-Technology-Work/swin_transformer/configs/swintransformer/swin_tiny.yaml（configs的绝对路径） --data_dir /home/xiangchengliu/Videos/Image-Measurement-Technology-Work/swin_transformer/dataset（datasets的绝对路径） --distribute False

    ```
    During training, you need to specify training parameters and data sets.
    The above path is an absolute path
    


## View training results

In order to view the training status, mindinsight must be installed

```pycon
pip install mindinsight
```
In order to view the loss changes during training, you can run

```pycon
mindinsight start --summary-base-dir=/home/xiangchengliu/Videos/Image-Measurement-Technology-Work/swin_transformer/mindinsight （mindinsight的绝对路径）--port=8091

```

## References


<details open markdown>
<summary> Paper </summary>

* Vision Transformer (ViT) - https://arxiv.org/abs/2010.11929
* Swin Transformer - https://arxiv.org/abs/2103.14030



</details>

## Used Algorithms

<details open markdown>
<summary> Supported algorithms </summary>

* Augmentation
    * [AutoAugment](https://arxiv.org/abs/1805.09501)
    * [RandAugment](https://arxiv.org/abs/1909.13719)
    * RandErasing (Cutout)
    * CutMix
    * MixUp
    * RandomResizeCrop
    * Color Jitter, Flip, etc
* Optimizer
    * AdamW
* LR Scheduler
    * Warmup Cosine Decay
* Regularization
    * Weight Decay
    * Label Smoothing
    * Stochastic Depth (depends on networks)
    * Dropout (depends on networks)
* Loss
    * Cross Entropy (w/ class weight and auxiliary logit support)

</details>


## License

This project follows the [Apache License 2.0](LICENSE.md) open-source license.

## Acknowledgement

MindCV is an open-source project jointly developed by the MindSpore team, Xidian University, and Xi'an Jiaotong University.
Sincere thanks to all participating researchers and developers for their hard work on this project.
We also acknowledge the computing resources provided by [OpenI](https://openi.pcl.ac.cn/).

# Image-Measurement-Technology-Work
Use mindsopre framework and mindcv library to complete the implementation of Swin Transformer algorithm
