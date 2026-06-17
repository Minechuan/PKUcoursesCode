# Installation

- We recommand using [Anaconda](https://www.anaconda.com/) to manage your python environments. Use the following command to create a new environment. Please change the 3.x bellow to your python version. Note that, we only test the code on python version higher than 3.7.
```bash
conda create -n hw2 python=3.x
conda activate hw2
```

- We recommand using [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/) to install dependent packages.

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

- Now you can install [pytorch](https://pytorch.org/get-started/previous-versions/) and other dependencies as below.
```bash
conda install pytorch torchvision cpuonly # remember to remove "-c pytorh"!

# tips: try "pip install xxx" first before "conda install xxx"
pip install tqdm
pip install opencv-python
pip install pillow
pip install tensorboardx
pip install tensorflow # for tensorboardx

# Uncomment the following line if you are using python >= 3.9
# Check https://github.com/pytorch/vision/issues/8536 for more information

# pip install numpy==1.26.4

```
The specific version of pytorch should make no difference for this assignment, since we only use some basic functions. You can also install the GPU version if you can access a GPU.

# CIFAR-10 

## Dataset
- Download cifar-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and save to `datasets`. Then unzip it.

```bash
cd datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
```

## Visualization
- Visualize the structure of your network
```bash
cd cifar-10
python network.py
cd ../experiments
tensorboard --logdir .
```

- Visualize and check the input data
```bash
cd cifar-10
python dataset.py
```

- Train network and visualize the curves
```bash
cd cifar-10
python train.py
cd ../experiments
tensorboard --logdir .
```

# Submission
- Compress the entire folder **except** `datasets` and `experiments`.

- Rename the compressed folder to **Name_ID.zip** and submit to [course.pku.edu.cn](https://course.pku.edu.cn/).

- The folder named `results` should be structed as follows.
```bash
  results
  ├── bn_loss.txt
  ├── Lenna.png 
  ├── Lenna_aug1.png 
  ├── Lenna_aug2.png 
  ├── Screenshot_LearningRate.png 
  └── Screenshot_YourBest.png

  ```

# Appendix 
We list some libraries that may help you solve this assignment.

- [TensorboardX](https://pytorch.org/docs/stable/tensorboard.html)
- [OpenCV-Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)
- [Torchvision.transforms](https://pytorch.org/vision/0.9/transforms.html)
