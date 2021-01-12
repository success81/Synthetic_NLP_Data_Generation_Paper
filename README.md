# Project Purpose
This repo is a project where I am exploring using GPT-2 to generate NLP observations for classification models.

## Background:
Classification Models use input data to predict the likelihood that the subsequent input data will fall into predetermined categories. To perform effective classifications, these models require large datasets for training. It is becoming common practice to utilize synthetic data to boost the performance of Machine Learning Models. It is reported that Shell is using synthetic data to build models to detect problems that rarely occur; for example Shell created synthetic data to help models to identify deteriorating oil lines.(Higginbotham, 2020) It is common practice for Machine Learning Practitioners to generate synthetic data by rotating, flipping, and cropping images to increase the volume of image data to train Convolutional Neural Networks(CNN). The purpose of this paper is to explore creating and utilizing synthetic NLP data to improve the performance of Natural Language Processing (NLP) Machine Learning Classification Models. In this paper I used a Yelp pizza restaurant reviews dataset and transfer learning to fine-tune a pre-trained GPT-2 Transformer Model to generate synthetic pizza reviews data. I then combined this synthetic data with the original genuine data to create a new joint dataset. For performance comparison purposes,  I built three baseline models on three separate datasets using the Multinomial Naive Bayes Classifier algorithm. The three datasets were: The Yelp Pizza Reviews Dataset (450 observations),  and a combined Yelp Pizza Reviews and Synthetic Yelp Reviews Dataset(11,380 observations). I used the scikit-learn train_test_split method on the genuine Yelp Pizza Reviews Dataset to develop a single ground truth test dataset. I then executed an analysis of the baseline models on the single ground truth test dataset to establish the following prediction performance metrics for each baseline model: precision, accuracy, recall, F1, and a confusion matrix. The combined Yelp Pizza Review Dataset outperformed the genuine Yelp Pizza Reviews Dataset on each of the performance metrics. I also pulled a "Car Repair" dataset to do comparison analysis.

## How to Navigate this Project
This project is organized into multiple Google Colab Notebooks. The code is commented within the notebooks so you can easily see what's going on.

## CRISP-DM 

The CRISP-DM planning document for this project can be found here:
https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/Crisp-DM.txt

## PROJECT WORK

# PHASE 1: DATA GATHERING/PREP SYNTHETIC DATA GENERATION
I First had to pull data from the Yelp Open Dataset I then downloaded the data(<b>Source</b>: 1,2,4). I then used GPT-2 to generate the synthetic data. (3). 

# PHASE 2: BUILDING THE MODELS
I then build the 

Sources
1. Pizza Reviews Genuine_Only Notebook: https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/JAN_FINAL_V2_Genuine_Pizza_Final.ipynb
2. Pizza Reviews Genuine and Synthetic Notebook: https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/JAN_FINAL_V2_Genuine_AND_GPT_Pizza_Final.ipynb
3. GPT Sample Notebook: https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/GPT_2_Sample_Notebook-2.ipynb
4. Car Repair Reviews https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/Car_SyntheticGenuine_Final_Capstone.ipynb

