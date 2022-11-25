This is an unofficial pytorch implementation of ["Complex-valued Iris Recognition Network"](https://arxiv.org/abs/2011.11198).

## 1. Install Requirements
Please install the following dependency in this project.
```bash
opencv-python
matplotlib
numpy
scipy
scikit
pytorch==1.11.0
torchvision==0.13.0
cplxmodule
```

## 2. Train Test
The experiments are performed on one machine with three GTX-1080Ti.
For training, please run the following code by specifying the path.
```python
 python train.py --base_path ../Data --mask_path ../Data -F Train_Test_List/train_nd_1s_LR.txt --batch_size 32 --dataset nd_1s
```
For testing, please run the following code by specifying the path.
```python
env CUDA_VISIBLE_DEVICES=0 python test.py --base_path <img_path> --mask_path <mask_path> --checkpoint <checkpoint_path>
```
