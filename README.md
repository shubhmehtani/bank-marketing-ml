# Deposit Prediction ML Model

## About

The goal of this model is to predict whether a customer will subscribe to a term deposit with the bank, based on their attributes. The target variable is binary, with a value of 1 indicating that the customer has subscribed to a term deposit, and a value of 0 indicating that they have not.

### Workflow:
1. Downloaded the dataset and evaluated it.
2. Performed Data Clearning and pre-processing and removed redundant features.
3. Used feature selection techniques like correlation, chi2 test, annova test, etc. for feature engineering.
4. Balanced the dataset using SMOTE-ENN since data is found to be HIGHLY imbalanced.
5. Normalized the training data using StandardScaler and testing data using the pretrained scaler to prevent data leakage.
6. Evaluated different perfomance metrics on multiple models like *Logistic Regression, KNNs, SVM, RandomForest, XGBoost, ExtraTrees, AdaBoost, LightGBM and MLP Classifier.*
7. Performed Hyperparameter tuning using RandomizedSearchCV on chosen model.
8. Achieved a final **Training Accuracy :  99.31% and Test Accuracy :  96.85%**


## Dataset

**Dataset Link:** [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

The banking dataset available on the UC Machine Learning Repository is a popular dataset used in the field of data science and machine learning. This dataset contains information on bank customers and their demographic, financial, and banking behavior attributes. The dataset consists of 45,211 rows and 17 columns.

Some of the attributes included in this dataset are age, job, marital status, education, housing loan status, and credit default status. The dataset also includes information on the last contact of the current campaign, as well as previous campaigns, such as the number of contacts made, the outcome of the previous campaign, and the time since the last contact.

## Analysis

| Models       | Training Accuracy | Testing Accuracy  |
| ------------- |:-------------:| -----:|
| Logistic Regression | 89.78 | 90.02 |
| KNN | 95.24   | 92.8 |
| SVM |  94.72 |   94.21 |
| Random Forest | 100 | 96.5 |
| AdaBoost | 95.2 |   95.4 |
| XGBoost | 98.27 |    96.72 |
| LightGBM | 97.11 | 96.54 |
| Extra Trees | 100 |   96.39 |
| MLP Classifier | 96.14 |   95.3  |
| XGBoost after Tuning | 99.31 |  96.85 |

## Final Result

**XGBoost Classifer** after hyperparemeter tuning gives best **testing accuracy: 96.85%**
