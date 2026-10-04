# Loan Approval Prediction

Comparing classical ML models (Decision Tree, KNN, SVM) on a real, messy Kaggle loan approval dataset.

## Problem
Predict whether a loan application will be approved, using applicant financial and demographic data.

## Data
Real Kaggle dataset — required cleaning: missing values (median/mode imputation), 
inconsistent category formatting, outliers.

## Approach
1. Established baseline (68.7%)
2. Compared Decision Tree, KNN, SVM (linear + RBF) using 5-fold cross-validation
3. Engineered features: loan-to-income ratio, total household income, income-per-dependent
4. Selected model based on both accuracy and stability (std across folds)

## Results
| Model | CV Mean Accuracy | Std |
|---|---|---|
| Baseline | 68.7% | - |
| Decision Tree | 79.2% | 0.011 |
| KNN (scaled) | 79.0% | 0.023 |
| SVM (linear) | 81.5% | 0.021 |
| SVM (RBF) | 80.9% | 0.014 |

## Key Learning
Feature scaling is critical for distance-based models — unscaled KNN scored 
*below* baseline. The highest-accuracy model isn't always the best choice — 
stability and explainability matter too.