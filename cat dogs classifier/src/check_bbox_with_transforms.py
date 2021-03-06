'''
Please provide index as input when running this script
Used to check if transforms are at all adequate.
'''

import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import albumentations as A
import albumentations.pytorch

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'height', 'width', 'xmin_coco', 'ymin_coco', 'xmax_coco', 'ymax_coco', 'label'])

# fetch random index
# idx = np.random.randint(df.shape[0])

# input
idx = int(input(f'Please provide a picture index between 0 and {df.shape[0]}: '))

# pull image data with random index OR provide index below
img_data = df.iloc[idx]

# you can provide index here
# img_data = df.iloc[477]

# image path
path = img_data['fname']

# open image
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# rectangle coordinates
x1 = img_data['xmin_coco']
y1 = img_data['ymin_coco']
x2 = img_data['xmax_coco'] - x1
y2 = img_data['ymax_coco'] - y1

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(image, bboxes, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bboxes
    x_min, x_max, y_min, y_max = int(x_min), int(
        x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        image,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return image


def visualize(image, bboxes, label, category_id_to_name):
    '''
    Visualize picture with the bbox.
    '''
    image = image.copy()
    for bboxes, category_id in zip(bboxes, label):
        class_name = category_id_to_name[category_id]
        image = visualize_bbox(image, bboxes, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


# has to be list of lists because there is for loop for the bboxes
bboxes = [[x1, y1, x2, y2]]
# fetch image class
label = [img_data.label]
# mapping used to display the class
category_id_to_name = {1: 'cat', 0: 'dog'}

# WITH TRANSFORMS

transform = A.Compose([
    # A.SmallestMaxSize(presize),
    # A.LongestMaxSize(presize),
    A.RandomSizedBBoxSafeCrop(config.presize, config.presize),
    # A.crops.transforms.CropAndPad(presize),
    # A.RandomCrop(crop, crop),
    # A.Normalize(),
    A.Rotate(limit=30),
    A.HorizontalFlip(p=0.5),
    A.Cutout(p=1.0),
    # albumentations.pytorch.ToTensorV2(),
],
    # bbox_params=A.BboxParams(format='coco'),
    bbox_params=A.BboxParams(format='coco', label_fields=['label']),
)

# transformed = transform(image=image, bboxes=bboxes)
transformed = transform(image=image,
                        bboxes=bboxes,
                        label=label)


visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['label'],
    category_id_to_name
)
