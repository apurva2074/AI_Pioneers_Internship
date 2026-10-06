# Machine Learning Data Preprocessing

## Overview

This project demonstrates the fundamentals of data preprocessing using Python, Pandas, NumPy, and Scikit-learn.

The project uses a diabetes dataset and covers data inspection, cleaning, missing-value handling, outlier analysis, feature selection, exploratory data analysis, and feature scaling.

## Dataset

The dataset contains 768 observations and 9 columns.

The target variable is `Outcome`:
- `0` = No diabetes
- `1` = Diabetes

## Preprocessing Steps

### 1. Data Inspection
- Examined the shape and structure of the dataset.
- Checked data types and identified the target variable.

### 2. Missing Values
Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were treated as invalid/missing measurements.

These values were converted to `NaN` and then imputed using mean or median depending on the observed distribution.

### 3. Duplicate Check
The dataset was checked for duplicate rows.

Result: 0 duplicate rows were found.

### 4. Outlier Analysis
The IQR method was used to identify potential outliers.

The detected outliers were retained because there was no evidence that they represented invalid measurements.

### 5. Feature Selection
`Outcome` was selected as the target variable.

The remaining eight numerical columns were retained as input features.

### 6. Exploratory Data Analysis
EDA included:
- Target distribution
- Feature distributions
- Glucose distribution by diabetes outcome
- Correlation heatmap

### 7. Categorical Encoding
No categorical encoding was required because all predictor variables were numerical.

### 8. Feature Scaling
Standardization using `StandardScaler` was applied.

The scaler was fitted only on the training data and then used to transform both training and testing data.

## Final Validation

After preprocessing:

- Training data: 614 rows × 8 features
- Testing data: 154 rows × 8 features
- Missing values: 0
- Duplicate rows: 0
- Scaled training mean: approximately 0
- Scaled training standard deviation: approximately 1

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook