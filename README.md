# Knowledge-Distillation

### 0. Overview
This repository contains a simple implementation of **[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf) (NIPS, 2014)** using PyTorch.
Knowledge Distillation is abbreviated as KD from below. A teacher network is the VGG19 with Batch Normalization network and a student model is a simple custom CNN network. The dataset used here is CIFAR dataset. 

Please note that the results may vary, depending on a set of hyper-parameters.


### 1. Quantitative Evaluation
#### 1) CIFAR10
| **Model** | **Top 1 Accuracy** | **Top 5 Accuracy** |
|:-------------:|:-------------:|:-----:|
| Teacher (VGG19 with BN) | 84.45% | 99.22% |
| Student (w/o KD) | 67.04% | 97.11% |
| **Student (with KD, Temp = 5)** | **79.72%** | **99.00%** |
| Student (with KD, Temp = 20) | 78.41% | 98.77% |

### 2. Accuracy and Loss during Training
#### a) CIFAR10

| Model | Accuracy | Loss |
|:-----:|:-----:|:-----:|
| Teacher | <img src = './results/plots/Teacher Model Accuracy using vgg CIFAR 10.png' width=600> | <img src = './results/plots/Teacher Model Loss using vgg CIFAR 10.png'> |
| Student (w/o KD) | <img src = './results/plots/Without Knowledge Distillation Accuracy using CIFAR 10.png'> | <img src = './results/plots/Without Knowledge Distillation Loss using CIFAR 10.png'> |
| Student (with KD, Temp = 5) | <img src = './results/plots/Knowledge Distillation Accuracy using CIFAR 10 and temp 5.png'> | <img src = './results/plots/Knowledge Distillation Loss using CIFAR 10 and temp 5.png'> |
| Student (with KD, Temp = 20) | <img src = './results/plots/Knowledge Distillation Accuracy using CIFAR 10 and temp 20.png'> | <img src = './results/plots/Knowledge Distillation Loss using CIFAR 10 and temp 20.png'> |


### 3. Run the Code
#### 1) Train

You should have a pre-trained teacher network in order to train a student network with KD. </br>
a) Train Teacher Network. 
```
python train.py --tto True
```

Then compare the performance between training the student network alone and distilling knowledge from teacher network to student network.
b) Train Student Network (without KD). 
```
python train.py
```

c) Train Student Network (with KD). 
```
python train.py --kd True --temp 5
```


#### 2) Evaluate
a) Evaluate Teacher Network 
```
python test.py --tto True
```

b) Evaluate Student Network (without KD). 
```
python test.py
```

c) Evaluate Student Network (with KD). 
```
python test.py --kd True
```


### Development Environment
```
- Ubuntu 18.04 LTS
- NVIDIA GFORCE RTX 3090
- CUDA 10.2
- torch 1.6.0
- torchvision 0.7.0
- etc
```
