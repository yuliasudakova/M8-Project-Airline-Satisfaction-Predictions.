# Airline Passenger Satisfaction Prediction ✈️

## Project Overview
This project analyzes airline passenger satisfaction and builds machine learning models to predict whether a passenger is **satisfied** or **neutral/dissatisfied** based on travel experience, service quality, and flight-related features.

The project is divided into three parts:

- **Part 1: Exploratory Data Analysis (EDA)**  
  Explore the dataset, analyze distributions, missing values, outliers, and important trends.

- **Part 2: Classification Modeling**  
  Train and evaluate classification models to predict passenger satisfaction.

- **Part 3: Streamlit App**  
  Present findings and model predictions in an interactive dashboard.

---

## Objective
The goal of this project is to:

- understand the factors that influence airline passenger satisfaction
- build classification models to predict satisfaction
- compare model performance
- present the results in an interactive Streamlit application

---

## Dataset
The dataset used in this project contains airline passenger information, including:

- demographic data
- travel type
- flight distance
- service ratings
- delay information
- satisfaction outcome

### Target Variable
- **satisfaction**
  - `satisfied`
  - `neutral or dissatisfied`
Part 1: Exploratory Data Analysis

In the EDA phase, the following steps were completed:

explored dataset structure and data types

analyzed distributions of key numerical and categorical features

checked and handled missing values

detected outliers

reviewed important passenger service variables

created visualizations to identify trends related to satisfaction

Key EDA Insights

Passenger satisfaction is strongly related to service quality features

Features such as seat comfort, inflight entertainment, cleanliness, and on-board service show clear differences between satisfied and dissatisfied passengers

Flight delays have a negative effect on passenger satisfaction

Travel experience and comfort appear to be major drivers of satisfaction

Part 2: Modeling

Several classification models were trained and evaluated to predict passenger satisfaction.

Models Tested

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest

Model Performance
Model	Accuracy
Logistic Regression	0.878
KNN	0.928
Random Forest	0.964
Conclusion from Modeling

The Random Forest model performed best and achieved the highest accuracy.
This suggests that passenger satisfaction depends on complex relationships between service ratings, travel characteristics, and flight delays that are captured well by a tree-based ensemble model.

Part 3: Streamlit App

A Streamlit app was built to present the project interactively.

App Features

View key findings from the exploratory data analysis

Review model performance results

Enter passenger information and predict satisfaction

Explore recommendations for improving passenger satisfaction

Run the App Locally
pip install -r requirements.txt
streamlit run air_app.py
Streamlit Deployment

The Streamlit app was deployed using Streamlit Cloud.

App Link

Add your deployed app link here:

[Streamlit App](PASTE_YOUR_STREAMLIT_LINK_HERE)
