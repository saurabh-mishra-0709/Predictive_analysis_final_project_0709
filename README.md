# AI Job Dataset Analysis

This project analyzes an AI job dataset using regression and classification techniques in Python. The objective is to study salary patterns, remote work opportunities, multicollinearity, influential observations, and compare different machine learning models.

---

## Project Objectives

* Predict salary using regression models
* Predict whether a job is fully remote using classification models
* Compare multiple machine learning techniques
* Detect multicollinearity using VIF
* Identify influential observations using Cook’s Distance
* Study interaction effects between experience and remote work
* Demonstrate simulation of least squares estimators

---

## Dataset Description

The dataset contains information about AI-related jobs.

### Variables Used

* `salary_usd` : Annual salary in USD
* `years_experience` : Number of years of experience
* `benefits_score` : Employee benefits score
* `experience_level` : Job experience level
* `education_required` : Minimum education requirement
* `company_size` : Size of the company
* `remote_ratio` : Percentage of remote work

---

## Data Preprocessing Steps

1. Removed missing values using `dropna()`
2. Selected only relevant variables
3. Created a binary target variable:

   * `is_remote = 1` if `remote_ratio = 100`
   * `is_remote = 0` otherwise
4. Converted categorical variables into dummy variables
5. Split the dataset into training and testing sets
6. Standardized numerical variables using `StandardScaler`

---

## Exploratory Data Analysis

The following visualizations were created:

* Salary vs Years of Experience scatter plot
* Salary vs Benefits Score scatter plot
* Salary distribution across experience levels using boxplots

---

## Models Used

### Regression Models

1. Multiple Linear Regression
2. K-Nearest Neighbors Regression
3. LASSO Regression
4. Ridge Regression
5. Elastic Net Regression

### Classification Models

1. Logistic Regression
2. K-Nearest Neighbors Classification
3. Gaussian Naive Bayes
4. Multinomial Logistic Regression

---

## Additional Statistical Analysis

* Variance Inflation Factor (VIF)
* Interaction Terms
* Cook’s Distance
* Simulation of Least Squares Estimators

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_curve, auc
```

---

## Project Workflow

1. Load and clean the dataset
2. Perform exploratory data analysis
3. Check multicollinearity using VIF
4. Build regression models for salary prediction
5. Build classification models for remote job prediction
6. Evaluate models using MSE, accuracy, ROC, and classification reports
7. Study interaction effects and influential observations
8. Summarize findings and compare model performance

---

## Key Findings

* Years of experience is one of the strongest predictors of salary
* Benefits score positively affects salary
* Remote job prediction can be achieved with reasonable accuracy
* LASSO helps identify the most important predictors
* Ridge and Elastic Net improve model stability
* Influential observations exist and can affect regression performance

---

## Author

Saurabh Mishra
