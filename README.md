# Santander Project 

 GitHub folder contains: 
 1. R code of project in ‘.R format’:  Santander_Final_Prtediction.R 
 2. Python code of project in ‘.py format’: Santander_Final_Prediction.py 
 3. Project report: Santander Customer Transaction Prediction Project.docx
 4. Predictions on test dataset in csv format:Final_Customer_Transaction_Predictions.csv
 
## Problem Statement 
 
In this problem goal is to predict the customer who will make the transaction and who will not irrespective of the money he/she spent in each transaction based upon the attributes given in the data 

Attributes: 
    Transaction_ID: object/character
    target - factor with only 2 levels i.e. "0" and "1": 0 stands for customer not making transaction and 1 means customer will make transaction
    201 variables - numeric/float type which do not contain the names but the values


### It is a classification Problem.
## All the steps implemented in this project
1. Data Pre-processing.
2. Data Visualization.
3. Exploratory data Analysis
3. Outlier Analysis.
4. Missing value Analysis.
5. Feature Selection.
 -  Feature Correlation analysis.
6. Splitting into Train and Validation Dataset.
8. Model Development
I.Logistic Regression without Class balance 
II. Decision Tree Regression 
III. Random Forest Regression 
IV. Gradient Boosting Method
9. Improve Accuracy 
a) Algorithm Tuning
b) Ensembles------XGBOOST For Regression
Finalize Model 
a) Predictions on validation dataset 
b) Create standalone model on entire training dataset 
c) Save model for later use
11. R code both in text format and also .R file
12. Python code
