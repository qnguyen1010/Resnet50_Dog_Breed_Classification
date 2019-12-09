# Resnet50_Dog_Breed_Classification

This notebook utilizes Resnet 50 in an Image Classification task. The dataset is downloaded from a Kaggle competition, using Kaggle API, input include username and key. 

https://www.kaggle.com/c/dog-breed-identification

Full code: Dog_breed_classification_QN.ipynb

## Data Pipeline

### 1. Load the labels.csv file into a dataframe using pandas and flow_from_dataframe

After this, we have a dataframe with 2 columns: **id** and **breed**

### 2. Load data into Train and Validation sets using ImageDataGenerator

**Importance** 

1. When using pretrained model, we should use preprocess_input imported directly from keras.applications as our preprocessing_function in the ImageDataGenerator. This is to prepare our images in accordance with the input format of the pretrained model. No need to use rescale = 1/255

2. Use **validation_split** in ImageDataGenerator to split train/val set. Subset: "training" for train set; "validation" for val set


## Building model

We use ResNet50, pretrained weights from Imagenet, and remove the top layers with include_top = False. Pretrained layers are not frozen, we continue to fine tune them.

## Train model

Trained for 10 epochs, val_acc reaches 0.6957

## Confusion Matrix and Classification Report on Validation set

See the code for more information
