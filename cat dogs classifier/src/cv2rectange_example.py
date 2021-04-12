import cv2
import pandas as pd

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=[1, 3, 4, 5, 6, 7])
# take any row
img_data = df.iloc[0]
# image path
path = img_data[0]
img = cv2.imread(path)
# rectangle coordinates
x1 = img_data[1]
y1 = img_data[2]
x2 = img_data[3]
y2 = img_data[4]
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
