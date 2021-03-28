# Short overview

This pet project aims to classify cars (or anything if you feed it images). Primarily done to try a few new things.

## problem:
- build a car classifier that can identify manufacturer, model, and color of a car using a provided image

## solution:
- end-to-end approach
- subset of kaggle dataset was used: https://www.kaggle.com/jonah28/myautoge-cars-data
- smaller subset used to train the model (overfits): https://www.kaggle.com/gobsan/small-car-dataset
- some manual data cleaning was done
- pretrained resnet18
- ROC AUC was used as a metric

## how to run:
- you can run the classifier directly from kaggle (turn on gpu): https://www.kaggle.com/gobsan/car-classifier-torch
- or you can open "5_inference.ipynb" file and add urls to url list to classify (there are a few urls already). Run all. You will need all the dependencies (Pipfile) and "checkpoint.pth" (in the same folder).

## notes:
- Please pay attention to '/' and '\\'. Make sure you use the right one for your OS.

## additional resources:
- quick baseline NB with fastai to test the new dataset created with "making_image_dataset.ipynb". Uses accuracy: https://www.kaggle.com/gobsan/car-class-baseline-fastai

## what was done
- tried downloader.py to check if it is enough to build an image dataset. To many duplicates. Didn't like the result
- download kaggle dataset
- used name generator to change all the names
- used making_image_dataset.ipynb to build new image dataset by sampling the original kaggle dataset
- removed some of the images from new dataset to leave cars only
- make car classifier with new dataset
