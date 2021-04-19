import os
from pathlib import Path
import models

# --------------------------------PATHS--------------------------------

# all is located in project_dir = ../
# execute scripts from ../src

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
CLASS_MODEL_OUTPUT = os.path.join(CHECKPOINT, 'class_checkpoint.pth')
LOCAL_MODEL_OUTPUT = os.path.join(CHECKPOINT, 'local_checkpoint.pth')

# --------------------------------TRAINING PARAMS--------------------------------
frac = 0.1  # this is the amount of original data used for training/validation/testing (low values will run the script faster)

model = models.resnet18(5)
# model = models.resnext50_32x4d(5)
# model = models.wide_resnet50_2(5)
# model = models.vgg(5)

lr = 0.01  # learning rate
momentum = 0.9  # momentum
weight_decay = 3e-3  # weight decay

# learning rate scheduler
step_size = 4  # after this many epochs we will mult our lr by gamma
gamma = 0.1  # lr multiplier every step_size epochs

# transforms
presize = 256  # initial resize of the image
crop = 224  # final crop of the image

# batch size
batch_size = 64

# n_epochs
frozen = 1  # number of epochs trained with only last layer's params available for training
unfrozen = 1  # all params are traininable

# tta (test time augmentation) region crop size
tta_crop = int(crop*0.9)
