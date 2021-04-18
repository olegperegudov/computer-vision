import pandas as pd
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt

import config
import dataset
import transforms
import localization_engine

'''
Will predict a bbox for an image and will display predicted and true bbox.
'''

# setting device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(filepath):
    '''
    This will load previously trained model.
    Please make sure you have the model available.
    '''
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

# checks if the model file exists
if not os.path.exists(os.path.join(config.CHECKPOINT, 'local_checkpoint.pth')):
    print(f'Please train the model with -localization_engine.py- first')

# load the model
model = load_checkpoint(os.path.join(config.CHECKPOINT, 'local_checkpoint.pth'))

# read dataframe
df = pd.read_csv(config.DF_PATH, usecols=['fname', 'xmin_alb', 'ymin_alb', 'xmax_alb', 'ymax_alb', 'label', 'kfold'])

dataset = dataset.localization_dataset(df, transforms.valid_transform_loc)

# input a picture index
idx = int(input(f'Please provide a picture index between 0 and {df.shape[0]}: '))
image, label = dataset[idx]

# predict new bbox
# need to remove 0 dim (batch dim)
image = torch.unsqueeze(image, 0)

# calculate outputs
outputs = model(image)
outputs = outputs.flatten()

# predicted coordinates for bbox
pred_bb = np.round(outputs[:-1], 4).tolist()  # <---- pred bb

# predicted bbox start
pred_start = (int(pred_bb[0]*config.crop), int(pred_bb[1]*config.crop))

# predicted bbox end
pred_end = (int(pred_bb[2]*config.crop), int(pred_bb[3]*config.crop))

# predicted (wont need it here)
pred_id = outputs[-1].long().item()

# need to permute tensor for visualization (make it so plt can visualize it)
# remove batch dim
image = torch.squeeze(image, 0)
# permute dims
image = np.array(image.permute(1, 2, 0))

# bbox = first 4 digits
bbox = np.round(label[:-1], 4).tolist()  # <---- true bb

# class = last digit
class_id = label[-1].long().item()

# true coordinates for bbox
start_point = (int(bbox[0]*config.crop), int(bbox[1]*config.crop))
end_point = (int(bbox[2]*config.crop), int(bbox[3]*config.crop))

# define a color and thickness
color = (0, 255, 0)
color2 = (255, 0, 0)

thickness = 2

# add 1st bbox, then add 2nd
image = cv2.rectangle(image, start_point, end_point, color, thickness)
image = cv2.rectangle(image, pred_start, pred_end, color2, thickness)

# show an image
plt.figure(figsize=(5, 5))
plt.imshow(np.array(image))
plt.show()

# GREEN = true, RED = predicted
