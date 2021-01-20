# Mushroom classifier (CNN)

The goal in this project is to train a CNN to classify pictures of mushrooms in over 800 different species.

## Introduction:

Mushrooms represent an important food source and they are used extensively in cooking, in many cuisines (notably Chinese, Korean, European, and Japanese). Furthermore, many people enjoy mushroom collection as a outdoor activity. However, this activity, entails some health risk, since some species that are poisoness look similar to edible specimens. In the example below I present such case of similarity between a delicious and edible mushroom *Macrolepiota mastoidea* and a *Amanita phaloides* which ingestion can lead to death.

Hence, it is important to discern what mushrooms are safe to pick.

In this notebook, I will train an algorithm that can aid us in classifying mushroom species.

<img src="figures/edible_vs_toxic.png" width="600"/> 

## Code and Resources Used:

**Python Version**: 3.7

**Packages used**: pandas, numpy, csv, bing_image_downloader, simple_image_download, pathlib, sys, re, datetime, os, time, glob, sklearn, matplotlib, seaborn, splitfolders, keras and tensorflow.

## 1. Scrape mushroom information and generate the picture dataset

   * **1.1** Get all the scientific names. 

   * **1.2** Use the scientific names to scrape pictures.

## 2. Prepare the picture dataset

   * **2.1**  First I will **re-size** all the pictures from the picture dataset to **350x350 pixels** 

   * **2.2** Convert all pictures into the **same format** (i.e. jpeg)

   * **2.3** Organize **train, test, and validation** image datasets into a consistent directory structure.

    ├── test
    │   ├── mushroom_name
    │   │   └── name_01.jpg
    │   └── ...
    │       └── name_50.jpg
    ├── train
    │   ├── mushroom_name
    │   │   └── name_01.jpg
    │   └── ...
    │       └── name_300.jpg
    └── validation
        ├── mushroom_name
        │   └── name_01.jpg
        └── ...
            └── name_50.jpg
            
## 3. Select CNN models for transfer learning

Three different models reported to have high accuracy were selected for transfer learning. See this [paper](https://arxiv.org/abs/1810.00736) for model performance comparission. 

 <img src="figures/image_classification_models.png" width="500"/> 
            
## 4. Create ImageDataGenerators and train the CNN model(s).

  * **4.1 Data Augmentation**: To expand the training dataset in order to improve the performance and ability of the model to generalize.
  
  See below an example on how data augmentation generates different variants of the same picture.
  
  <img src="figures/data_augmentation.png" width="400"/> 


* **4.2 Xception(2016) Model Training - Trainable params: 27,946,080**

<img src="figures/Xception_Training.png" width="600"/> 

We observe a better performance on the validation set compared to the training set. A possible explanation is that the validation set may be easier than the training set. [See this article](https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/). After inspecting the picture set obtained from google images, I observed pictures duplicated in both sets. Hence, some data cleaning by removing duplicate pictures may help by making the model to generalize better.


* **4.3 Xception Performance Evaluation**

Xception model -> accuracy: 70.30%, loss: 1.02

Example Single Picture Evaluation:


<img src="figures/nomo.jpeg" width="200"/>  

    * Top 1 Prediction: With 48.6% probability is a picture of Amanita muscaria.
    * Top 2 Prediction: With 18.7% probability is a picture of Amanita hongoi.
    * Top 3 Prediction: With 18.1% probability is a picture of Amanita heterochroma.
    * Top 4 Prediction: With  6.5% probability is a picture of Amanita parcivolvata.
    * Top 5 Prediction: With  4.9% probability is a picture of Amanita xanthocephala.

<img src="figures/niscalos.jpeg" width="200"/> 

    * Top 1 Prediction: With 89.2% probability is a picture of Lactarius deliciosus.
    * Top 2 Prediction: With  5.6% probability is a picture of Lactarius salmonicolor.
    * Top 3 Prediction: With  2.6% probability is a picture of Lactarius semisanguifluus.
    * Top 4 Prediction: With  1.5% probability is a picture of Lactarius.
    * Top 5 Prediction: With  1.0% probability is a picture of Lactarius sanguifluus.

Next, I will improve the picture dataset, I include more mushroom species and train EfficientNetB7 (2020), a model containing more trainable parameters (Ongoing...▶).


* **4.4 EfficientNetB7 (2020) Model Training - Trainable params: 71.964.414, picture_size=(600x600)**


* **4.5 EfficientNetB7 Performance Evaluation**
