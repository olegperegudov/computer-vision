# This is a dog vs cat classification and localization solution

## Short overview of the solution:

- there are 2 engines (training scripts) - classification and classification with localization
- classification (classification_engine.py) will only classify images as dog or cat image and will not predict bboxes at all
- classification with localization (localization_engine.py) will train a regression type model that will attempt to predict bboxes for cats' and dogs' faces.
- I've used pretrained resnet18 for both tasks. It should be stated that this architecture is not in any way optimal and one will get much better results with other architectures, such as ssd, yolo, faster rccn etc. I used resnet18 just for fun and see if it works at all for a regressiont ask. I have also tried other models: vgg, wide resnet, resnet 50, alexnet.
- I have got ~99% accuracy (train/valid/test) and ~65% mIoU with ~18 epochs, where ~3 epochs were with frozen weights (except for the last layers) and the rest of the epochs were done with all the weights available for training.

# Project structure:

- working dir with project name. Inside there are:
- '**checkpoint**' folder for trained models (will be created autimatically when you train your 1st model)
- '**data**' folder with all the data - images and txt files
- '**input**' folder for dataframe. This folder and the dataframe will be created ones you run 'df_preparation.py' from 'src' folder
- '**notebooks**' folder. It has some extra files used during this project. You can still run 'localization_engine.ipynb' and 'classification_engine.ipynb' from it. They are almost identical to their '...py' analogs in 'src'. This folder is not needed for anything
- '**src**' folder is the main folder to run all the scripts inside. 'src' folder is set as 'cwd' for all the scripts.
- inside '**src**' there are the following files:
- '**check_bbox_with_transforms**.py' - add/change transforms inside, display the result with bbox
- '**check_predicted_bbox**.py' - used to display a predicted bbox once you have a model. Will run full pipeline for a single image.
- '**classification_engine**.py' - will train and create classification model/checkpoint to predict class
- '**config**.py' - config file with (almost) all the parameters and paths
- '**dataset**.py' - custom dataset class
- '**df_preparation**.py' - this is the 1st script you run. It will 'walk' the 'data' folder and create the dataframe to use later
- '**localization_engine**.py' - will train and create localization model/checkpoint tol predict bboxes 
- '**models**.py' - different models I used for training. There are quite a few.
- '**show_picture**.py' - simple script to display an image from df using indexing
- '**transforms**.py' - all the transforms for all the engines

# how to run the project:

- you can run it as is at kaggle. Just turn on gpu and run all. Should take ~9 mins. Or...
- clone or dl repo from git
- you will need all the data unpacked inside 'data' folder. Can be downloaded here: https://www.kaggle.com/gobsan/cats-vs-dogs-bbox
- install all the dependencies (poetry)
- from within the 'src' folder, run 'df_preparation.py'. It will create 'input folder' and place a df inside it
- run 'classification_engine.py' for classification model or 'localization_engine.py' for localization model. Models will be places inside checkpoint folder (will be created if doesn't exist)
- please note that both engines will run with only 0.1 total data and for 2 epochs total. This is to make sure everything works as intended. If all is good, please go to config file and change 'frac' to 1 (to use all available data for training, validation and testing) and change 'frozen' and 'unfrozen' number of epochs to whichever number you want (I used 3 and 15 respectively). With GPU it will take ~10 mins with all the data and ~20 total epochs
- after you have the 'localization' model, you can run 'check_predicted_bbox.py' give it image index and see the predicted results.
- you can also check any picture in the dataset providing the index with 'show_picture.py'
- you can play around with 'check_bbox_with_transforms.py' by setting any transformations inside it (I used albumentations module) and running the script. At one point it helped me a lot to debug one really sneaky problem.
