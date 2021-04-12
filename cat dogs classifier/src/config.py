import os
from pathlib import Path

# all is located in project_dir = ../

# ../
ROOT = Path.cwd().parent

# image data @ ../data
DATA = os.path.join(ROOT, 'data')

# input dfs @ ../input
INPUT = os.path.join(ROOT, 'input')

# main df we will work with @ ../input
DF_PATH = os.path.join(INPUT, 'train_data.csv')

# all pretrained models @ ../models
CHECKPOINT = os.path.join(ROOT, 'checkpoint')

# path to save trained model @ ../models
MODEL_OUTPUT = os.path.join(CHECKPOINT, 'checkpoint.pth')
