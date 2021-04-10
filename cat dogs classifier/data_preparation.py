import os
import pandas as pd

# we will populate this predefined table
df = pd.DataFrame(columns=['fname', 'info', 'xmin',
                           'ymin', 'xmax', 'ymax', 'label'])

# setting paths
ROOT = os.getcwd()
DATA = os.path.join(ROOT, 'data')

# walking image folder and constructing the df
image_fnames = []
image_info = []

for _, _, fnames in os.walk(DATA):
    for fname in fnames:
        extension = fname.split(".")[-1].lower()
        # we will need full path + fname later
        if extension == 'jpg':
            image_fnames.append(fname)
        # extract data from the txt file
        if extension == "txt":
            FILE_PATH = os.path.join(DATA, fname)
            with open(FILE_PATH) as file:
                txt = file.readlines()
                image_info.append(txt)

# pushing the data collected with os.walk to the df
df['fname'] = [(DATA+fname) for fname in image_fnames]
df['info'] = image_info

# parsing the string data. Making the df pretty
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []
label_list = []

for row in df['info']:
    for txt in row:
        splitted_text = txt.split(' ')
        xmin_list.append(int(splitted_text[1]))
        ymin_list.append(int(splitted_text[2]))
        xmax_list.append(int(splitted_text[3]))
        ymax_list.append(int(splitted_text[4]))
        label_list.append(int(splitted_text[0]))

df.xmin = xmin_list
df.ymin = ymin_list
df.xmax = xmax_list
df.ymax = ymax_list
df.label = label_list

# set cats:1 and dogs:0
df.label = df.label.map({1: 1, 2: 0})

# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# just some info for the final pring
total_cats = df[df.label == 1].shape[0]
total_dogs = df[df.label == 0].shape[0]

# safe the new df
fname = 'train_data'
df.to_csv(fname + '.csv')
print(f'File {fname}.csv with {total_cats} cats and {total_dogs} dogs created at working dir.')
