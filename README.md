# Mushroom classifier (CNN)

The goal in this project is to train a CNN to classify pictures of mushrooms in over 800 different species.

## Summary:

Mushrooms represent an important food source and they are used extensively in cooking, in many cuisines (notably Chinese, Korean, European, and Japanese). Furthermore, many people enjoy mushroom collection as a outdoor activity. However, this activity, entails some health risk, since some species that are poisoness look similar to edible specimens. In the example below I present such case of similarity between a delicious and edible mushroom *Macrolepiota mastoidea* and a *Amanita phaloides* which ingestion can lead to death.

Hence, it is important to discern what mushrooms are safe to pick.

In this notebook, I will train an algorithm that can aid us in classifying mushroom species.

<img src="figures/edible_vs_toxic.png" width="600"/> 

## Code and Resources Used:

**Python Version**: 3.7

**Packages**: pandas, numpy, csv, re, datetime, os, sklearn, matplotlib, seaborn, plotly.express, splitfolders, keras and tensorflow.

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
    │       └── name_01.jpg
    ├── train
    │   ├── mushroom_name
    │   │   └── name_01.jpg
    │   └── ...
    │       └── name_01.jpg
    └── validation
        ├── mushroom_name
        │   └── name_01.jpg
        └── ...
            └── name_01.jpg
            
## 3. Create ImageDataGenerators and train the CNN model(s) Xception, Inception V4 and ResNeXt50 + CBAM

  * **3.1 Data Augmentation**: To expand the training dataset in order to improve the performance and ability of the model to generalize.
  
  See below an example on how data augmentation generates different variants of the same picture.
  <img src="figures/data_augmentation.png" width="300"/> 


