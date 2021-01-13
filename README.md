# Project Purpose
This repo is a project where I am exploring using GPT-2 to generate NLP observations for classification models.

# Background:
Classification Models use input data to predict the likelihood that the subsequent input data will fall into predetermined categories. To perform effective classifications, these models require large datasets for training. It is becoming common practice to utilize synthetic data to boost the performance of Machine Learning Models. It is reported that Shell is using synthetic data to build models to detect problems that rarely occur; for example Shell created synthetic data to help models to identify deteriorating oil lines.It is common practice for Machine Learning Practitioners to generate synthetic data by rotating, flipping, and cropping images to increase the volume of image data to train Convolutional Neural Networks(CNN). The purpose of this paper is to explore creating and utilizing synthetic NLP data to improve the performance of Natural Language Processing (NLP) Machine Learning Classification Models. In this paper I used a Yelp pizza restaurant reviews dataset and transfer learning to fine-tune a pre-trained GPT-2 Transformer Model to generate synthetic pizza reviews data. I then combined this synthetic data with the original genuine data to create a new joint dataset. For performance comparison purposes, I built two baseline models on two separate datasets using the Multinomial Naive Bayes Classifier algorithm. The two datasets were: The Yelp Pizza Reviews Dataset (450 observations),  and a combined Yelp Pizza Reviews and Synthetic Yelp Reviews Dataset(11,380 observations). I created a single ground truth test dataset from the original Yelp dataset. I then executed an analysis of the baseline models on the single ground truth test dataset to establish the following prediction performance metrics for each baseline model: precision, accuracy, recall, F1, and a confusion matrix. The combined Yelp Pizza Review Dataset outperformed the genuine Yelp Pizza Reviews Dataset on each of the performance metrics. I also pulled a Car Repair dataset to do comparison analysis.

# Project Assumptions
By adding synthetic data to genuine training data the performance of the Classification Models will be improved


# Project Walkthrough

## 2. Methodology:

## 2.1 Introduction
For the research conducted in this paper I used the GPT-2 transformer model and Yelp open dataset pizza reviews to create the synthetic data. 

## 2.1 GPT-2
Developed by OpenAI, GPT-2 is a large-scale transformer-based language model that is pre-trained on a large corpus of text: 8 million high-quality webpages. The objective of GPT-2 is to predict the next word given all of the previous words within some text. The GPT-2 model can be trained with an additional custom dataset using a method called transfer learning to produce more relevant text.

## 2.2 Yelp Open Dataset Reviews
The Yelp Open Dataset contains anonymized reviews on various businesses and services (Yelp). For this paper I created a subset of data of pizza restaurant reviews. Within this subset of data I divided the ratings into “Positive” and “Negative”. Ratings that were 4 or 5 stars were categorized as “Positive”. Ratings that were 1 or 2 stars were categorized as “Negative”. For this paper my Negative dataset contained 225 observations and the Positive dataset also contained 225 observations. 

<p align="center">
  <img width="1000" height="500" src="https://user-images.githubusercontent.com/9426716/104496217-b08e9b00-55a6-11eb-9f8d-932b25909730.png">
</p>

## 2.3 Technical Approach

The intent of the research in this paper is to train two GPT-2 models on a small subset of Positive and Negative Yelp Pizza Reviews data. I will then use the two GPT-2 models to produce synthetic Positive and Negative review datasets. I will finally combine the new synthetic datasets with the genuine dataset and fit a classification model to this dataset that will have the ability to determine negative and positive sentiment of pizza restaurant reviews.

## 2.3.1 Generating Synthetic Review Data

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496008-6b6a6900-55a6-11eb-9b8f-23c9fbe8480d.png">
</p>

My first task was to create two GPT-2 models and train one of them on genuine negative Yelp pizza review data and the other model on genuine positive Yelp pizza review data. I then had these models generate synthetic negative and positive review data that was combined into one a single dataset.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104495863-4249d880-55a6-11eb-9e38-b14db1924b29.png">
</p>

## 2.3.2 Synthetic Review Generation
I chose the 355 million parameter GPT-2 model to build my two models. I used Google Colabs as my development notebook. 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104497032-c486cc80-55a7-11eb-9dcf-6420f6c48a9e.png">
</p>

I used the GPT2.Generate method to generate synthetic reviews. On average, my responses were 70 words long. 

<p align="center">
  <img width="1000" height="500" src="https://user-images.githubusercontent.com/9426716/104495471-cf406200-55a5-11eb-86ca-214bd8f48fac.png">
</p>

## 2.3.3 GPT-2 Text Generation

When generating synthetic reviews I wanted to ensure that the responses expanded on the genuine data and produced responses that were a strong representation of the genuine data. So when I wrote the prefix prompts I used words that were heavily represented in the genuine datasets.  I wrote a Python Function that organized the genuine dataset corpus into trigrams (3 word consecutive combinations), bigrams (2 word combinations) and words. This function also provides a count and numerically sorts the occurrences of these words and combinations. 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104495367-b041d000-55a5-11eb-87a8-d30a3c79fb8f.png">
</p>

2.3.4 Genuine and Synthetic Dataset Concatenation 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104495259-8a1c3000-55a5-11eb-9822-369689989eb1.png">
</p>

After I created the synthetic Positive and Negative datasets I concatenated them with the genuine Negative and Positive Datasets. 

## 2.3.5 Performance Testing

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104495089-4b867580-55a5-11eb-91bd-be64f1e50e11.png">
</p>

To ensure there was a fair and equal analysis of the performance metrics, I used the scikit-learn train_test_split method to establish a single ground truth test set consisting of 198 observations derived from a totally separate dataset from the Yelp Open Dataset. I then built two baseline models on two datasets using the Multinomial Naive Bayes Classifier algorithm. The two datasets were: The genuine Yelp Pizza Reviews Dataset (450 observations) and the combined Genuine and Synthetic Yelp Reviews Dataset(11,380 observations).

## 2.3.6 Performance Testing Results


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496763-65c15300-55a7-11eb-9ae4-d08b02b607d7.png">
</p>



<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496947-a7ea9480-55a7-11eb-88d6-c7fcea1d57bb.png">
</p>

These were the results of the pizza and car review models.


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496864-89849900-55a7-11eb-820a-593940d9785c.png">
</p>



These were the performance percentage increases of the pizza and car models.


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496481-0d8a5100-55a7-11eb-80fb-098de0b35029.png">
</p>

These were the results when I added 1000 synthetic observations to the car repair genuine data at a time.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/9426716/104496608-327ec400-55a7-11eb-8912-f28d419c537f.png">
</p>

These were the results of LSTM models.


## 3. Concluding Remarks:


In conclusion, the Synthetic and Genuine Dataset performed well in all performance metrics. This technique has the possibility of allowing organizations and businesses to build high performing NLP classification models without the high cost associated with large scale data acquisition. There are opportunities in exploring this technique on datasets with a larger observation count. There are also opportunities in exploring GPT-2 prompt design to better guide the GPT-2 model in generating relevant text. This is an exciting Machine Learning Technique that I feel deserves further exploration.






# How to Navigate this Project Code
This project is organized into multiple Google Colab Notebooks. The code is commented within the notebooks and PY files so you can easily see what's going on. The files are already linked to files in this repo. 

# CRISP-DM 

The CRISP-DM planning document for this project can be found here:
https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/Crisp-DM.txt

# PROJECT WORK CODE

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
