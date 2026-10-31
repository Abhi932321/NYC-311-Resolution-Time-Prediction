# NYC 311 Service Request Resolution Time Prediction

A Predictive Analytics project that uses machine learning to classify NYC 311 service requests into Fast, Medium, and Slow resolution categories.

## Project Overview

This project analyzes NYC 311 service request data to understand resolution patterns and predict the resolution category of service requests using five machine learning algorithms.

The dataset contains 100,000 sampled records from the selected historical data period.

## Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Linear Support Vector Classifier (LinearSVC)

A Dummy Classifier was also used as a baseline.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

- Data collection and cleaning
- Missing value handling
- Feature engineering
- Exploratory Data Analysis (EDA)
- Categorical encoding and numerical scaling
- Chronological training and testing split
- Machine learning model training
- Model evaluation and comparison

## Model Performance

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 79.50% | 78.78% | 75.06% |
| Decision Tree | 78.04% | 71.09% | 70.65% |
| KNN | 79.72% | 73.07% | 72.64% |
| Gaussian Naive Bayes | 77.47% | 76.41% | 72.20% |
| LinearSVC | 80.65% | 78.53% | 75.59% |

LinearSVC achieved the highest accuracy, while Logistic Regression achieved the highest balanced accuracy.

## Dataset

Source: [NYC Open Data – 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)

The analysis uses a selected historical period from February 27, 2024, to March 31, 2024.

The dataset is not included in this repository. Download the data separately from the official source.

## Resolution Categories

- Fast: Up to 24 hours
- Medium: More than 24 hours and up to 168 hours
- Slow: More than 168 hours

These are project-defined categories, not official NYC service standards.

## Important Note

The analysis uses requests with valid closure dates and non-negative resolution times. Predictions concern historical closed requests; administrative closure does not necessarily confirm that the underlying issue was fully resolved.

## Author

Abhishek Kumar  
B.Tech CSE  
Lovely Professional University
