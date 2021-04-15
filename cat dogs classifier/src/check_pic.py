import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import albumentations as A
import albumentations.pytorch

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'height', 'width',
                                          'xmin_coco', 'ymin_coco', 'xmax_coco', 'ymax_coco', 'label'])

# problematic pictures
# 66, 477, 1818, 2167
img_data = df.iloc[477]
# image path
path = img_data['fname']
# open image
image = plt.imread(path)
plt.imshow(image)
plt.show()

