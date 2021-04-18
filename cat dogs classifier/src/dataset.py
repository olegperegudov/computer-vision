from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import cv2

# dataset class


class dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, index):
        image = Image.open(self.df.fname[index]).convert('RGB')
        image = np.array(image)

        label = torch.tensor(self.df.label[index]).long()

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations['image']

        return image, label


class localization_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, index):
        image = cv2.imread(self.df.fname[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_id = self.df.label[index]

        xmin = self.df.xmin_alb[index]
        ymin = self.df.ymin_alb[index]
        xmax = self.df.xmax_alb[index]
        ymax = self.df.ymax_alb[index]

        bboxes = [[xmin, ymin, xmax, ymax, class_id]]

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes)

            image = transformed['image']
            bboxes = transformed['bboxes']

        label = torch.tensor(bboxes).flatten()

        return image, label
