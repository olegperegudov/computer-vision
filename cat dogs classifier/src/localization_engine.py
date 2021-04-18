'''
This is CLASSIFICATION AND LOCALIZATION engine.
It does the training, validation and testing.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import numpy as np
import os

import pandas as pd
import time

import models
import config
import dataset
import localization_transforms

# starting time
start = time.time()

# setting seed
torch.manual_seed(0)
np.random.seed(0)

# setting device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: this will only use a fraction of total available data. Set it to '1' in config to use all data
df = pd.read_csv(config.DF_PATH, usecols=['fname',
                                          'xmin_alb', 'ymin_alb', 'xmax_alb', 'ymax_alb',
                                          'label', 'kfold']).sample(frac=config.frac).reset_index(drop=True)

# create dataframes using folds we have got with "df_preparation.py"
train_df = df[df.kfold.isin([0, 1, 2])].reset_index(drop=True)
valid_df = df[df.kfold == 3].reset_index(drop=True)
test_df = df[df.kfold == 4].reset_index(drop=True)

# create dataset
train_dataset = dataset.localization_dataset(train_df, localization_transforms.train_transform)
valid_dataset = dataset.localization_dataset(valid_df, localization_transforms.valid_transform)
test_dataset = dataset.localization_dataset(test_df, localization_transforms.test_transform)

# create loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size)

# check if all is good with the shapes of the loaders
assert next(iter(train_loader))[0].shape[0] == config.batch_size, 'Something is wrong with train loader'
assert next(iter(train_loader))[0].shape[1] == 3, 'Something is wrong with train loader'
assert next(iter(train_loader))[0].shape[2] == config.crop, 'Something is wrong with train loader'
assert next(iter(train_loader))[0].shape[3] == config.crop, 'Something is wrong with train loader'

assert next(iter(valid_loader))[0].shape[0] == config.batch_size, 'Something is wrong with valid loader'
assert next(iter(valid_loader))[0].shape[1] == 3, 'Something is wrong with valid loader'
assert next(iter(valid_loader))[0].shape[2] == config.crop, 'Something is wrong with valid loader'
assert next(iter(valid_loader))[0].shape[3] == config.crop, 'Something is wrong with valid loader'

assert next(iter(test_loader))[0].shape[0] == config.batch_size, 'Something is wrong with test loader'
assert next(iter(test_loader))[0].shape[1] == 3, 'Something is wrong with test loader'
assert next(iter(test_loader))[0].shape[2] == config.crop, 'Something is wrong with test loader'
assert next(iter(test_loader))[0].shape[3] == config.crop, 'Something is wrong with test loader'

'''
# displayig the data (looks this way because of normalization)
batch_tensor = next(iter(train_loader))[0][:6,...]
grid_img = torchvision.utils.make_grid(batch_tensor, nrow=3)

# grid_img.shape
plt.figure(figsize=(16,6))
plt.imshow(grid_img.permute(1, 2, 0));
'''

# setting up a model
model = config.model.to(device)
# loss
criterion = nn.MSELoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min')


def train_model(n_epochs=1,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler):

    '''
    Main function for training.
    '''

    # start total time
    total_time = time.time()

    for epoch in range(n_epochs):
        # go train mode
        model.train()
        # start epoch time
        t0 = time.time()
        # these will be used for metric calculations
        correct_on_epoch = 0
        total_num_images = 0
        epoch_loss = 0

        for batch, (images, labels) in enumerate(train_loader):
            # move images and labels to gpu, if available
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # keep track of total images in one epoch
            total_num_images += labels.size(0)
            # clear grads before forward pass
            optimizer.zero_grad()
            outputs = model(images.float())
            # we only need labels from (1,5) size tensor
            preds = outputs[:, -1].round()
            # calculate the batch training loss
            loss = criterion(labels.float(), outputs.float())
            # keep track of total epoch loss
            epoch_loss += loss
            # correct on bacth
            correct_on_batch = (preds == labels[:, -1]).sum().item()  # train acc
            # correct on epoch
            correct_on_epoch += correct_on_batch  # train acc
            # backward pass and step
            loss.backward()
            optimizer.step()

        # train acc/loss
        train_epoch_acc = round((correct_on_epoch/total_num_images), 4)  # train acc
        train_avg_epoch_loss = round(float(epoch_loss/len(train_loader)), 4)
        # valid acc/loss
        valid_avg_epoch_loss, valid_epoch_accuracy, mean_iou = test_model(model, valid_loader)

        # epoch time
        epoch_time = round(time.time() - t0)
        # for reduce on plateau LR
        lr_scheduler.step(valid_avg_epoch_loss)

        print(f'epoch: [{epoch+1}/{n_epochs}] | train loss: {train_avg_epoch_loss} | train acc: {train_epoch_acc} | valid loss: {valid_avg_epoch_loss} | valid acc: {valid_epoch_accuracy} | iou: {mean_iou} | time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s')

    return model


def test_model(model, test_loader):

    '''
    Main function for testing.
    '''
    model.eval()

    correct_on_epoch = 0
    total_num_images = 0
    epoch_loss = 0
    epoch_iou = []

    all_batch_acc = []

    with torch.no_grad():

        for batch, (images, labels) in enumerate(test_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            true_bb = labels[:, :-1]

            total_num_images += images.size(0)

            outputs = model(images)
            preds = outputs[:, -1].round()
            pred_bb = outputs[:, :-1]

            loss = criterion(outputs, labels)
            epoch_loss += loss

            correct_on_epoch += (preds == labels[:, -1]).sum().item()

            # batch iou
            batch_iou = iou(true_bb, pred_bb)
            epoch_iou.append(batch_iou)

    test_epoch_accuracy = round((correct_on_epoch/total_num_images), 4)
    test_avg_epoch_loss = round(float(epoch_loss/len(test_loader)), 4)
    mean_iou = np.round(np.mean(epoch_iou), 4)

    return test_avg_epoch_loss, test_epoch_accuracy, mean_iou


def iou(true_bb, pred_bb):

    '''
    Main function for calculating Intersection over Union
    '''

    batch_iou = []

    for idx, (true, pred) in enumerate(zip(true_bb, pred_bb)):

        pred = torch.clip(pred, min=0.0, max=1.0).to('cpu')
        true = torch.clip(true, min=0.0, max=1.0).to('cpu')

        xmin_t, ymin_t, xmax_t, ymax_t = true
        xmin_p, ymin_p, xmax_p, ymax_p = pred

        xmin_intersect = np.maximum(xmin_t, xmin_p)
        ymin_intersect = np.maximum(ymin_t, ymin_p)
        xmax_intersect = np.minimum(xmax_t, xmax_p)
        ymax_intersect = np.minimum(ymax_t, ymax_p)

        if xmin_intersect < xmax_intersect and ymin_intersect < ymax_intersect:

            intersection_area = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
            union_area = (xmax_t-xmin_t)*(ymax_t-ymin_t)+(xmax_p-xmin_p)*(ymax_p-ymin_p)-intersection_area + 1e-6

            assert intersection_area > 0, 'intersection area cat be < 0'
            assert union_area > 0, 'union area cant be < 0'

            iou = intersection_area / union_area
            batch_iou.append(iou)

        else:
            batch_iou.append(0.0)

    return np.round(np.mean(batch_iou), 4)


# unfreeze all the params for training
def unfreeze(model=model):
    '''
    This finction will unfreeze previously freezed params of our model so all the model's params will be available for training.
    '''
    for param in model.parameters():
        param.requires_grad = True
    return model


if __name__ == "__main__":
    # train a model with freezed params for a few epochs, unfreeze all params and train some more
    print(f'================')
    print(f'Started training with frozen params...')
    print(f'================')
    train_model(config.frozen)
    unfreeze()
    print(f'================')
    print(f'Started training with unfrozen params...')
    print(f'================')
    train_model(config.unfrozen)
    print(f'================')
    print('Training done.')
    print(f'================')
    print(f'Started testing...')
    print(f'================')

    # testing with train data
    _, train_acc, train_iou = test_model(model, train_loader)
    print(f'train acc: {train_acc} | train iou: {train_iou}')

    # testing with test data
    _, test_acc, test_iou = test_model(model, test_loader)
    print(f'test acc: {test_acc} | test iou: {test_iou}')

    print(f'================')
    print(f'Testing done.')
    # total run time
    total_time = time.time() - start
    print(f'================')

    # check for path to save the model @ ../checkpoint
    if not os.path.exists(config.CHECKPOINT):
        os.mkdir(config.CHECKPOINT)

    # we can save pur model here for future inference
    checkpoint = {'model': model,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, config.LOCAL_MODEL_OUTPUT)

    print(f'Total time: {total_time//60:.0f}m {total_time%60:.0f}s')
