# Maxup实现

文章https://arxiv.org/pdf/2002.09024v1.pdf
基于RESNET34模型

##### 所需：

Python 3.7.7
torch 1.4.0
torchvision 0.5.0
tqdm 4.45.0
albumentations 0.4.5
scikit-learn 0.22.2.post1
opencv-python 4.1.1.26

```pip install -r requirements.txt```

##### 数据集：

ImageNet 2012 DataSets

图片置于以下目录

```./data/imagenette2-320/train```

##### 代码内容：

配置：

```python config.py```

数据集调用：

```python dataset.py```

模型：

```python model.py```

数据增强：

```python cutmix.py```

模型训练：

```python train.py```

参数：

* ```cutmix``` - 使用cutmix;
* ```m``` - Maxup的Cutmix放大倍数;
* ```device``` - 使用CPU或GPU;
* ```epochs``` - 检查点数;
* ```pretrained_weights``` - 预制模型的天平路径.

失真计算：

```python maxup_loss.py```

质量评估：

```python eval.py```

参数：

* ```weights``` - 模型秤的路径;
* ```device``` - 使用CPU或GPU.

##### 结果：

```Resnet34 + MaxUp+CutMix, m=4```

/maxup_implementation-master/train.py

1842it [13:23, 2.29it/s]

Epoch 0 -> Train Loss: 2.3157

1842it [25:30, 1.20it/s]

Epoch 1 -> Train Loss: 2.3146

1842it [17:42, 1.73it/s]

Epoch 2 -> Train Loss: 2.3168

1842it [14:02, 2.19it/s]

Epoch 3 -> Train Loss: 2.3184

1842it [14:22, 2.14it/s]

Epoch 4 -> Train Loss: 2.3155

Process finished with exit code 0

质量评估

|                                          | Accuracy, % |
| :--------------------------------------: | :---------: |
|                 Resnet34                 |    96.83    |
|            Resnet34 + CutMix             |    97.01    |
|       Resnet34 + MaxUp+CutMix, m=4       |    97.68    |
| Resnet34 + MaxUp+CutMix, m=4, fine-tuned |    98.23    |

模型重量

 * Resnet34: ```./result/Base_exp/weights.pth```

 * Resnet34 + CutMix (m=1): ```./result/Cutmix_exp/weights.pth```

 * Resnet34 + MaxUp+CutMix, m=4: ```./result/Cutmix_maxup_4_exp/weights.pth```

 * Resnet34 + MaxUp+CutMix, m=4 (fine-tuned): ```./result/Cutmix_maxup_4_exp/weights.pth```

