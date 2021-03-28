# Project overview:

- Dataset used: https://www.kaggle.com/gobsan/st-george-or-not
- Dataset description: binary classification of ~6000 pics with St. George.
- This NB attempts to classify a given dataset with Saint George using Pytorch and a pretrained resnet50 model. No extra work was done to the dataset (more images or cleaning).

This is what was done:

1. Prep work included using Pandas for path and label collection of all data.
2. The data was split into train, valid and test datasets.
3. Some minor augmentations added to the data.
4. The model was 1st trained with all but the last layer frozen for a couple of epochs, then all params were unfrozen and the whole model trained again.
5. TTA was used on test dataset.
6. I could get ~6.5% error rate with 3 frozen and ~10 unfozen epochs.

The model shows signs of overfitting the train set, but valid and test set error rates are good indicators of the model working properly.

## How to run:

1. By far the easiest would be "copy & edit" + "run all" it directly on kaggle.com - https://www.kaggle.com/gobsan/enter-ideas-george-torch
This way it will have all the needed dependencies and dataset preinstalled.
Just turn on GPU acceleration and run the whole thing. It will take ~ 15 min if you run it as is.

2. You can also run inference NB only. With no training required, you can check it at: https://www.kaggle.com/gobsan/enter-ideas-george-torch-inference
Same as before - "copy & edit" + "run all". It shouldn't take more than a few seconds for result to pop up. Can run CPU or GPU. You can try to classify a few images yourself providing URL.

3. You can try and run it locally with Jupyter NB if you have GPU available. Train a new model with classifier file or use inference file to see how the final model works.

ps. There is also a quick and dirty 10 min NB with fastai which gives pretty good results.
https://www.kaggle.com/gobsan/homework-enter-ideas-fastai
