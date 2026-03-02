# Industrial Anomaly Detection Using Machine Learning

## Project Overview

This project focuses on multi-class anomaly detection in industrial generator fuel-consumption data. The objective is to identify abnormal operational patterns using supervised machine learning techniques.

## Problem

The dataset contained:
- Severe class imbalance
- Noisy operational logs
- Cluster-specific distribution shifts
- Rare but critical anomaly classes (<3%)

Standard accuracy metrics were misleading due to imbalance.

## Approach

- Data preprocessing and feature engineering
- SMOTE-ENN for class balancing
- Comparison of global vs cluster-specific models
- Evaluation using Macro-F1 and AUC-ROC
- SHAP explainability analysis for interpretability

## Key Results

- Significant improvement in minority-class recall
- Improved anomaly detection consistency
- Identified domain-shift issues across clusters

## Technologies Used

- Python
- Scikit-learn
- Imbalanced-learn
- SHAP
- Pandas
- NumPy

## Future Improvements

- Domain adaptation techniques
- Fairness analysis
- Model compression for deployment
