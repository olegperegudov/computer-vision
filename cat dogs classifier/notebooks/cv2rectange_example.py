import cv2
import pandas as pd
import numpy as np

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'height', 'width',
                                          'xmin', 'ymin', 'xmax', 'ymax', 'label'])

# chekc random picture
idx = np.random.randint(df.shape[0])
# take any row
img_data = df.iloc[idx]
# image path
path = img_data['fname']
img = cv2.imread(path)
# rectangle coordinates
x1 = img_data['xmin']
y1 = img_data['ymin']
x2 = img_data['xmax']
y2 = img_data['ymax']
# extra params
color = (255, 0, 0)
thickness = 2
# convert rectangle coords to 2 points
start = (x1, y1)
end = (x2, y2)
# save image
image = cv2.rectangle(img, start, end, color, thickness)
# draw image
cv2.imshow('image', image)
cv2.waitKey(0)
