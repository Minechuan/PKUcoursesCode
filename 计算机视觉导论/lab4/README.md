# Installation

The installation is almost the same as HM2. You can **directly use the HM2 environment and pip install the extra packages listed below** or create a new environment by following the steps below:

- We recommend using [Anaconda](https://www.anaconda.com/) to manage your python environments. Use the following command to create a new environment.
```bash
conda create -n hw4 python=3.7 # use python=3.8 on Mac
conda activate hw4
```

- We recommend using [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/) to install dependent packages.

```bash
# pip
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

- Now you can install [pytorch](https://pytorch.org/get-started/previous-versions/) and other dependencies as below. Choose the version that fits your machine. The specific version of pytorch should make no functional difference for this assignment, since we only use some basic functions. You can also install the GPU version if you can access a GPU.
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly

# tips: always try "pip install xxx" first before "conda install xxx"
pip install opencv-python
pip install pillow
pip install tensorboardx
pip install matplotlib # new for HW4
pip install imageio # new for HW4
pip install h5py # new for HW4
```
You can also install the GPU version if you can access a GPU.

# ShapeNetPart for PointNet

## Dataset
- You can download both datasets from the course mirror: https://disk.pku.edu.cn/link/AA7806F129368D4534B86FCCA0C3CF5D76
- Alternatively, download the ShapeNetPart `v0_normal` dataset from its official source:
```bash
wget https://huggingface.co/datasets/wangps/shapenet_segmentation/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

Then unzip it:
```bash
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

## Data Configuration
Open `PointNet/utils.py`, and modify the dataset path:
```
dataset = "YOUR_PATH/shapenetcore_partanno_segmentation_benchmark_v0_normal"
```


## Visualization

- Train network and visualize the curves
```bash
cd PointNet
python train_classification.py -d 256
cd ../exps
tensorboard --logdir .
```


# Self-Attention Captioning

This is the new lightweight task for HW4. It uses the same COCO image features and captions as the RNN task, but replaces the recurrent layer with causal self-attention. Please complete the self-attention functions and module in `Attention/attention.py`.

Run the public numerical checks:

```bash
cd Attention
python check_attention.py
python check_transformer_caption.py
```

The checks cover causal mask construction, scaled dot-product attention, multi-head self-attention, and a tiny Transformer captioning forward / backward / sampling case.

After finishing the attention code, you can train a small Transformer captioning model on the same data as the RNN demo:

```bash
cd Attention
python train_transformer_caption.py
```

This script overfits 50 COCO captioning samples and saves loss and prediction figures under `Attention/results/`. It does not require training a real CLIP, LLM, VLM, or large Transformer.


# RNN
- You can download both datasets from the course mirror: https://disk.pku.edu.cn/link/AA7806F129368D4534B86FCCA0C3CF5D76
- Alternatively, download and unzip the CS231n COCO captioning dataset:
```bash
wget https://cs231n.stanford.edu/coco_captioning.zip
unzip coco_captioning.zip
```
- Then set the variable `BASE_DIR='your_path_to_coco_captioning_folder'` in `RNN/utils/coco_utils.py`. We recommend using an absolute path. The Attention training script uses the same loader and the same `BASE_DIR`.


# Submission
- Compress your code and results using our provided script `pack.py` and submit the generated zip file to the course website.
- Before running `pack.py`, make sure the following required files exist:

```text
PointNet/model.py
PointNet/results/0.png
PointNet/results/1.png
PointNet/results/2.png
PointNet/results/3.png
PointNet/results/classification_256.png
PointNet/results/classification_1024.png
PointNet/results/segmentation.png
Attention/attention.py
Attention/transformer_caption.py
RNN/rnn.py
RNN/rnn_layers.py
RNN/results/pred_train_0.png
RNN/results/pred_train_1.png
RNN/results/pred_val_0.png
RNN/results/pred_val_1.png
RNN/results/rnn_loss_history.png
RNN/results/single_rnn_layer.npy
RNN/results/rnn.npy
```

The Transformer captioning result files under `Attention/results/` are optional and will be included if they exist.

# Appendix and Acknowledgement
We list some libraries that may help you solve this assignment.

- [TensorboardX](https://pytorch.org/docs/stable/tensorboard.html)
- [OpenCV-Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)
- [Torchvision.transforms](https://pytorch.org/vision/0.9/transforms.html)

Our code is inspired by [PointNet-Pytorch](https://github.com/fxia22/pointnet.pytorch), [detection-torchvision](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and cs231n.
