# Project Purpose
This repo is a project where I am exploring using GPT-2 to generate NLP observations for classification models.

<center>
<img src="https://user-images.githubusercontent.com/9426716/104441775-20c6fd80-5562-11eb-8d9e-bd3afc7342fb.jpg" class="center" width="300" height="300">
</center>

# Background:
Classification Models use input data to predict the likelihood that the subsequent input data will fall into predetermined categories. To perform effective classifications, these models require large datasets for training. It is becoming common practice to utilize synthetic data to boost the performance of Machine Learning Models. It is reported that Shell is using synthetic data to build models to detect problems that rarely occur; for example Shell created synthetic data to help models to identify deteriorating oil lines.It is common practice for Machine Learning Practitioners to generate synthetic data by rotating, flipping, and cropping images to increase the volume of image data to train Convolutional Neural Networks(CNN). The purpose of this paper is to explore creating and utilizing synthetic NLP data to improve the performance of Natural Language Processing (NLP) Machine Learning Classification Models. In this paper I used a Yelp pizza restaurant reviews dataset and transfer learning to fine-tune a pre-trained GPT-2 Transformer Model to generate synthetic pizza reviews data. I then combined this synthetic data with the original genuine data to create a new joint dataset. For performance comparison purposes, I built two baseline models on two separate datasets using the Multinomial Naive Bayes Classifier algorithm. The two datasets were: The Yelp Pizza Reviews Dataset (450 observations),  and a combined Yelp Pizza Reviews and Synthetic Yelp Reviews Dataset(11,380 observations). I created a single ground truth test dataset from the original Yelp dataset. I then executed an analysis of the baseline models on the single ground truth test dataset to establish the following prediction performance metrics for each baseline model: precision, accuracy, recall, F1, and a confusion matrix. The combined Yelp Pizza Review Dataset outperformed the genuine Yelp Pizza Reviews Dataset on each of the performance metrics. I also pulled a Car Repair dataset to do comparison analysis.

# Project Assumptions
By adding synthetic data to genuine training data the performance of the Classification Models will be improved

# How to Navigate this Project
This project is organized into multiple Google Colab Notebooks. The code is commented within the notebooks and PY files so you can easily see what's going on. The files are already linked to files in this repo. 

# CRISP-DM 

The CRISP-DM planning document for this project can be found here:
https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/Crisp-DM.txt

# PROJECT WORK

## PHASE 1: DATA GATHERING/PREP SYNTHETIC DATA GENERATION
I first had to pull data from the Yelp Open Dataset(https://www.yelp.com/dataset). I then loaded the data into a Google Colabs Notebook<b>(References: 1,3)</b>. I then used GPT-2 to generate the synthetic data<b>(Reference: 9)</b>. 

## PHASE 2: BUILDING THE MODELS
I then built a baseline Multinomial Model for the Pizza Reviews and Car Repair Reviews. The Baseline models had no synthetic data. I then added the synthetic data 
to the genuine data and to form new Models. <b>(Reference: 1,2,3,4)</b> I then built a LSTM model with genuine data only and a mix of genuine and synthetic data for the Car repair and Pizza Reviews. <b>(Reference: 5,6,7,8)</b>

## PHASE 3: MODEL PERFORMANCE TESTING
The performance of the models overall improved with the addition of synthetic data to the genuine data. The results can be seen here. <b>(Reference: 10)</b>

### Performance Testing Code

Pizza Review Baseline / Pizza Review Improved             <b>(Reference:1,1A,2,2A)</b>

Car Repair Baseline / Car Repair Review Improved          <b>(Reference:3,3A,4,4A)</b>

LSTM Models                                               <b>(Reference:7,7A,8,8A)</b>

# Conclusion
Through the testing of multiple models, it is proven that adding synthetic data improves the performance of Classification Models. 

# Additional Information
<b>Towards Data Science Article I wrote on this technique:</b> 
https://towardsdatascience.com/the-magic-of-synthetic-data-using-artificial-intelligence-to-train-artificial-intelligence-with-cebe0d50a411?sk=5f8437e52422a7933c86e28e8c01e797

<b>Paper I wrote on this technique:</b>
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/GPT-2%20NLP%20Synthetic%20Data%20ML%20Paper%20JAN3-2.pdf

# References

1: Pizza Reviews Genuine Only Notebook:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/JAN_FINAL_V2_GENUINE_Pizza_FinalZ.ipynb

1A: Pizza Reviews Genuine Only PY File:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/jan_final_v2_genuine_pizza_finalz.py

2: Pizza Reviews Genuine and Synthetic:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/JAN_FINAL_V2_Genuine_AND_GPT_Pizza_FinalZ.ipynb

2A: Pizza Reviews Genuine and Synthetic PY File: https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/jan_final_v2_genuine_and_gpt_pizza_finalz.py

3: Car Repair Genuine:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/Car_Genuine_Final_CapstoneZ.ipynb

3A: Car Repair Genuine PY File:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/car_genuine_final_capstonez.py

4: Car Repair Genuine/Synthetic:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/Car_GenuineSynthetic_Final_CapstoneZ.ipynb

4A:Car Repair Genuine/Synthetic Python:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/car_genuinesynthetic_final_capstonez.py

5: LSTM Pizza Genuine: 
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Final_LSTM_CAPSTONE_Pizza_GenuineZ.ipynb

5A: LSTM Pizza Genuine PY file:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/final_lstm_capstone_pizza_genuinez.py

6: LSTM Pizza Genuine and Synthetic Notebook:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Final_Pizza_LSTM_CAPSTONE_Guide_CombinedZ.ipynb

6A: LSTM Pizza Genuine and Synthetic PY:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/final_pizza_lstm_capstone_guide_combinedz.py

7: LSTM Car Repair Genuine:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/Final_Car_LSTM_CAPSTONE_Guide_GenuineZ.ipynb

7A: LSTM Car Repair Genuine PY:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/final_car_lstm_capstone_guide_genuinez.py

8: LSTM Car Repair Genuine and Synthetic:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/Final_Car_LSTM_CAPSTONE_Guide_CombinedZ.ipynb

8A: LSTM Car Repair Genuine and Syntnetic PY:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/Car_repair/final_car_lstm_capstone_guide_combinedz.py

9:GPT Sample Notebook:                                  
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/GPT_2_Sample_Notebook-2.ipynb

10:Performance Results:
https://github.com/success81/Synthetic_NLP_Data_Generation_Paper/blob/main/BASELINE%20MODEL%20PERFORMANCE%20REVIEWS.pdf
