import cv2
import os
import glob
import torch
import json

from PIL import Image
from torch.utils.data import Dataset


class Imagenette(Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 size=320,
                 valid=False,
                 transform=None):
        # print(root)
        img_path = os.path.join(root, mode)
        classes_path = glob.glob(''.join([img_path, '/*']))
        # print(classes_path)

        '''label2Idx 是数据集映射使用'''
        with open(os.path.join(root, 'label2Idx.json'), 'r') as file:
            label2Idx = json.load(file)
        '''路径全改为/'''
        classes_path2 = []
        for class_path in classes_path:
            class_path2 = class_path.replace('\\', '/')
            classes_path2.append(class_path2)
            
        if valid is True:
            self.img_data = [[os.path.join(class_path, f), label2Idx[class_path.split('/')[-1]]]
                             for class_path in classes_path2
                             for f in os.listdir(class_path) if 'val' in f]
        else:
            self.img_data = [[os.path.join(class_path, f), label2Idx[class_path.split('/')[-1]]]
                             for class_path in classes_path2
                             for f in os.listdir(class_path) if 'val' not in f]
        self.num_classes = len(label2Idx)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.img_data)
    '''加载数据集中各数据属性'''
    def __getitem__(self, index):
        img_file, label = self.img_data[index]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (self.size, self.size))
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label] = 1.
        return img, label_onehot
