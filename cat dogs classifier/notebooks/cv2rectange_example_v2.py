import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'height', 'width',
                                          'xmin_coco', 'ymin_coco', 'xmax_coco', 'ymax_coco', 'label'])

# chekc random picture
idx = np.random.randint(df.shape[0])
# take any row
img_data = df.iloc[idx]
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


def visualize(image, bboxes, category_ids, category_id_to_name):
    image = image.copy()
    for bboxes, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        image = visualize_bbox(image, bboxes, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


bboxes = [[x1, y1, x2, y2]]
# print(f'bboxes: {bboxes}')
category_ids = [img_data.label]
# print(f'category_ids: {category_ids}')
category_id_to_name = {1: 'cat', 0: 'dog'}
# print(f'category_id_to_name: {category_id_to_name[category_ids[0]]}')

# visualize
visualize(image,
          bboxes,
          category_ids,
          category_id_to_name)
