# MaxViT
pytorch实现MaxViT，可以在ImageNet或自己的数据集上训练，支持apex混合精度，中断后自动加载权重训练，以及各种图像增强技术

## MaxViT官方实现代码（TensorFlow版本）：https://github.com/google-research/maxvit
## MaxViT网络代码（未实现训练代码）：https://github.com/ChristophReich1996/MaxViT

Unofficial **PyTorch** reimplementation of the
paper [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf)
by Zhengzhong Tu et al. (Google Research).

<p align="center">
  <img src="maxvit.png"  alt="1" width = 674px height = 306px >
</p>

Figure taken from [paper](https://arxiv.org/pdf/2204.01697.pdf).


## Pretrained MaxViT Checkpoints

We have provided a list of results and checkpoints as follows:

|     Name      | resolution |    Top1 Acc.  |    #Params   |  FLOPs   
|    ----------     |  ---------|    ------    |    ------   | ------  
|    MaxViT-T      |  224x224  |   83.62%   |    31M    |  5.6B    | 
|    MaxViT-T     |  384x384   |   85.24%   |    31M    | 17.7B    | 
|    MaxViT-T     |  512x512   |   85.72%   |   31M    | 33.7B    | 
|    MaxViT-S     |  224x224   |  84.45%   |    69M    |  11.7B    | 
|    MaxViT-S     |  384x384   |   85.74%   |    69M    | 36.1B    | 
|    MaxViT-S     |  512x512   |    86.19%   |   69M    | 67.6B    | 
|    MaxViT-B     |  224x224   |    84.95%   |   119M    | 24.2B    | 
|    MaxViT-B     |  384x384   |    86.34%   |   119M    | 74.2B    | 
|    MaxViT-B     |  512x512   |    86.66%   |   119M    | 138.5B    | 
|    MaxViT-L     |  224x224   |    85.17%   |   212M    | 43.9B    | 
|    MaxViT-L     |  384x384   |    86.40%   |   212M    | 133.1B    | 
|    MaxViT-L     |  512x512   |    86.70%   |   212M    | 245.4B    |


# Install
- Create a conda virtual environment and activate it:

```bash
conda create -n maxvit python=3.7 -y
conda activate maxvit
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm`:

```bash
pip install timm
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
  
# Train for scratch
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

For example, to train `ffc_base` with 8 GPU on a single node for 300 epochs, run:

`max_vit_tiny`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/max_vit_tiny.yaml --data-path <imagenet-path> --batch-size 128
```

`max_vit_small`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/max_vit_small.yaml --data-path <imagenet-path> --batch-size 128
```

`max_vit_base`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/max_vit_base.yaml --data-path <imagenet-path> --batch-size 128
```

`max_vit_large`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/max_vit_large.yaml --data-path <imagenet-path> --batch-size 128
```
