import os
# from pathlib import Path
import pandas as pd

df = pd.DataFrame(columns=['id', 'fname', 'class',
                           'xmin', 'ymin', 'xmax', 'ymax'])


ROOT = os.getcwd()
DATA = os.path.join(ROOT, 'data')

print('started walk')
for _, _, fnames in os.walk(DATA):
    for fname in fnames:
        # print(fname)
        extension = fname.split(".")[-1].lower()
        if extension == "txt":
            FILE_PATH = os.path.join(DATA, 'fname')
            with open(fname) as file:
                txt = file.readlines()
                # print(txt)

# Path.setcwd(DATA)
# print(ROOT)
# print(os.listdir(DATA)[:5])
