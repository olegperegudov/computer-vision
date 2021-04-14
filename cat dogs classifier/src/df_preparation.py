import os
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import config

# this function will create extra column in our df with kfold values


def create_folds(data, n_splits=5, shuffle=True):
    """Makes a copy of df with extra column at the end - "kfolds"
       and set fold's number in it.
    Args:
        n_splits (int): number of KFold splits
        shuffle (bool, optional): whether shuffle the df on splits
    """
    # we'll go for stratifiedKFold since there is an imbalance in classes
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=1)

    # fetch labels
    y = df.label.values

    # loop to set value for kfold column
    for idx, (_, valid) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid, 'kfold'] = idx

    df['kfold'] = df['kfold'].astype('int')

    return df


if __name__ == "__main__":

    # we will populate this predefined table later
    df = pd.DataFrame(columns=['fname', 'height', 'width', 'colors', 'info', 
                               'xmin_coco', 'ymin_coco', 'xmax_coco', 'ymax_coco',
                               'xmin_alb', 'ymin_alb', 'xmax_alb', 'ymax_alb', 
                               'label'])

    # walking image folder and constructing the df
    image_fnames = []
    image_info = []
    image_height = []
    image_width = []
    image_colors = []

    for _, _, fnames in os.walk(config.DATA):
        for fname in fnames:
            extension = fname.split(".")[-1].lower()
            FILE_PATH = os.path.join(config.DATA, fname)
            # we will need full path + fname later
            if extension == 'jpg':
                img = cv2.imread(FILE_PATH)
                h, w, c = img.shape
                image_height.append(h)
                image_width.append(w)
                image_colors.append(c)
                image_fnames.append(fname)
            # extract data from the txt file
            if extension == "txt":
                with open(FILE_PATH) as file:
                    txt = file.readlines()
                    image_info.append(txt)

    # pushing the data collected with os.walk to the df
    df['fname'] = [(os.path.join(config.DATA, fname))
                   for fname in image_fnames]
    df['info'] = image_info
    df['height'] = image_height
    df['width'] = image_width
    df['colors'] = image_colors

    # creating empty future columns' data
    xmin_coco_list = []
    ymin_coco_list = []
    xmax_coco_list = []
    ymax_coco_list = []
    label_list = []

    # parsing the string data
    for row in df['info']:
        for txt in row:
            splitted_text = txt.split(' ')
            xmin_coco_list.append(int(splitted_text[1]))
            ymin_coco_list.append(int(splitted_text[2]))
            xmax_coco_list.append(int(splitted_text[3]))
            ymax_coco_list.append(int(splitted_text[4]))
            label_list.append(int(splitted_text[0]))

    # fill in the empty columns with data
    df.xmin_coco = xmin_coco_list
    df.ymin_coco = ymin_coco_list
    df.xmax_coco = xmax_coco_list
    df.ymax_coco = ymax_coco_list

    # normalize bbox coordinates
    df.xmin_alb = df.xmin_coco/df.width
    df.ymin_alb = df.ymin_coco/df.height
    df.xmax_alb = df.xmax_coco/df.width
    df.ymax_alb = df.ymax_coco/df.height

    df.label = label_list

    # set cats:1 and dogs:0
    df.label = df.label.map({1: 1, 2: 0})

    # just some info for the final print
    total_cats = df[df.label == 1].shape[0]
    total_dogs = df[df.label == 0].shape[0]

    # create shuffeled kfolds for validation (if we decide to use it)
    df = create_folds(data=df)

    # safe the new df
    fname = 'train_data'
    df.to_csv(os.path.join(config.INPUT, fname + '.csv'))
    print(
        f"-- File {fname}.csv with {total_cats} cats and {total_dogs} dogs created in 'input' dir. --")

    # check the distribution of classes. Sum should be the same -> same number of cats in each fold
    # print(
        # f"Cats distribution between folds: {df.groupby(['kfold']).label.sum()}")
