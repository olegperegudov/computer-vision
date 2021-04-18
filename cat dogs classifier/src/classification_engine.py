'''
This file does the training, validation and testing for CLASSIFICATION part of the problem.
There is no bbox prediction here.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import numpy as np
import os

import pandas as pd
import time
import ttach as tta

import models
import config
import dataset
import classification_transforms

# starting time
start = time.time()

# setting seed
torch.manual_seed(0)
np.random.seed(0)

# setting device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: this will only use a fraction of total available data. Set it to '1' in config to use all data
df = pd.read_csv(config.DF_PATH).sample(frac=config.frac)

# create dataframes using folds we have got with "df_preparation.py"
train_df = df[df.kfold.isin([0, 1, 2])].reset_index(drop=True)
valid_df = df[df.kfold == 3].reset_index(drop=True)
test_df = df[df.kfold == 4].reset_index(drop=True)

# create datasets
train_dataset = dataset.dataset(train_df, classification_transforms.train_transform)
valid_dataset = dataset.dataset(valid_df, classification_transforms.valid_transform)
test_dataset = dataset.dataset(test_df, classification_transforms.test_transform)

# create loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True)

'''
# displayig the data (looks this way because of normalization)
batch_tensor = next(iter(train_loader))[0][:10, ...]
grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)

# grid_img.shape
plt.figure(figsize=(16, 6))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
'''

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

# setting up a model
model = models.resnet18(2).to(device)  # good
# model = models.vgg(2).to(device) # good
# model = models.alexnet(2).to(device) # bad

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config.step_size, gamma=config.gamma)


def train_model(n_epochs=1,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler):
    '''
    Main function to train the model.
    '''

    total_time = time.time()

    for epoch in range(n_epochs):

        # train mode
        model.train()
        # start training time
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
            total_num_images += images.size(0)
            # clear grads before forward pass
            optimizer.zero_grad()
            # outputs
            outputs = model(images)
            # get a prediction from outputs
            _, preds = torch.max(outputs, 1)
            # calculate the bacth training loss
            loss = criterion(outputs, labels)
            # keep trach of total epoch loss
            epoch_loss += loss
            # keep track of correct predictions during each batch and add it to epoch's correct
            correct_on_epoch += (preds == labels).sum().item()  # train acc
            # backward pass and step
            loss.backward()
            optimizer.step()
        # epoch train accuracy
        train_epoch_acc = round((correct_on_epoch/total_num_images), 4)
        # epoch train loss
        train_avg_epoch_loss = round(float(epoch_loss/len(train_loader)), 4)
        # epoch test loss and accuracy
        valid_avg_epoch_loss, valid_epoch_accuracy = test_model(
            model, valid_loader)
        # total epoch time
        epoch_time = round(time.time() - t0)
        # learning scheduler step
        lr_scheduler.step()

        print(f'epoch: [{epoch+1}/{n_epochs}] | train loss: {train_avg_epoch_loss} | train acc: {train_epoch_acc} | valid loss: {valid_avg_epoch_loss} | valid acc: {valid_epoch_accuracy} | time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s')

    return model


def test_model(model, test_loader):
    '''
    Main function to test the model.
    '''
    model.eval()

    correct_on_epoch = 0
    total_num_images = 0
    epoch_loss = 0

    all_batch_acc = []

    with torch.no_grad():

        for batch, (images, labels) in enumerate(test_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            total_num_images += images.size(0)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            epoch_loss += loss

            correct_on_epoch += (preds == labels).sum().item()

    test_epoch_accuracy = round((correct_on_epoch/total_num_images), 4)
    test_avg_epoch_loss = round(float(epoch_loss/len(test_loader)), 4)

    return test_avg_epoch_loss, test_epoch_accuracy


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
    _, train_acc = test_model(model, train_loader)
    print(f'train set acc: {train_acc}')

    # testing with test data
    _, test_acc = test_model(model, test_loader)
    print(f'test set acc: {test_acc}')

    # this is a 5x tta with test data
    # wrapping a model
    tta_model = tta.ClassificationTTAWrapper(
        model, tta.aliases.five_crop_transform(config.tta_crop, config.tta_crop))
    # creating dataset
    tta_dataset = dataset.dataset(
        valid_df, classification_transforms.tta_transform)
    # creating the loder
    tta_loader = DataLoader(tta_dataset, batch_size=1, shuffle=False)
    # testing
    _, tta_acc = test_model(tta_model, tta_loader)

    print(f'TTA acc: {tta_acc}')
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
    torch.save(checkpoint, config.CLASS_MODEL_OUTPUT)

    print(f'Total time: {total_time//60:.0f}m {total_time%60:.0f}s')
