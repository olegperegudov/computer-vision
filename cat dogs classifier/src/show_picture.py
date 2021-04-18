'''
This script will show picture (w/o bbox) from input/df provided you feed it index.
'''

import pandas as pd
import matplotlib.pyplot as plt

import config

# dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'height', 'width',
                                          'xmin_coco', 'ymin_coco', 'xmax_coco', 'ymax_coco', 'label'])


index = input(f'Please provide a picture index between 0 and {df.shape[0]}: ')
# problematic pictures
# 66, 477, 1818, 2167
img_data = df.iloc[int(index)]
# image path
path = img_data['fname']
# open image
image = plt.imread(path)
plt.imshow(image)
plt.show()
