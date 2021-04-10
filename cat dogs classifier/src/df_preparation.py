import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def create_folds(data, n_splits=5, shuffle=True):
    """Makes a copy of df with extra column at the end - "kfolds"
       and set fold's number in it.
    Args:
        n_splits (int): number of KFold splits
        shuffle (bool, optional): whether shuffle the df on splits
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=1)

    # fetch labels
    y = df.label.values

    # loop to set value for kfold column
    for idx, (_, valid) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid, 'kfold'] = idx

    df['kfold'] = df['kfold'].astype('int')

    return df


if __name__ == "__main__":

    # we will populate this predefined table
    df = pd.DataFrame(columns=['fname', 'info', 'xmin',
                               'ymin', 'xmax', 'ymax', 'label'])

    # setting paths
    ROOT = Path.cwd().parent
    # image data files
    DATA = os.path.join(ROOT, 'data')
    # dataframes for training
    INPUT = os.path.join(ROOT, 'input')

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
    df['fname'] = [(os.path.join(DATA, fname)) for fname in image_fnames]
    df['info'] = image_info

    # creating empty future columns' data
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    label_list = []

    # parsing the string data
    for row in df['info']:
        for txt in row:
            splitted_text = txt.split(' ')
            xmin_list.append(int(splitted_text[1]))
            ymin_list.append(int(splitted_text[2]))
            xmax_list.append(int(splitted_text[3]))
            ymax_list.append(int(splitted_text[4]))
            label_list.append(int(splitted_text[0]))

    # fill in the empty columns with data
    df.xmin = xmin_list
    df.ymin = ymin_list
    df.xmax = xmax_list
    df.ymax = ymax_list
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
    df.to_csv(os.path.join(INPUT, fname + '.csv'))
    print(
        f'File {fname}.csv with {total_cats} cats and {total_dogs} dogs created at working dir.')

    # check the distribution of classes. Sum should be the same -> same number of cats in each fold
    # print(df.groupby(['kfold']).label.sum())
