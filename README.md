# Explainable Credit Risk Prediction

## Project Overview

This research project investigates the prediction of credit-card default using interpretable and machine-learning classification models. It evaluates predictive performance, probability calibration, model explainability, and the trade-offs involved in identifying potential defaults.

The project is designed as a reproducible Data Science Research Capstone, culminating in a formal research whitepaper.

## Research Question

How effectively can interpretable machine-learning models predict credit-card default, and how do their predictive performance, probability calibration, and explainability compare with a conventional logistic-regression baseline?

## Research Objectives

* Examine the dataset's structure, quality, and default-class distribution.
* Conduct exploratory data analysis and document meaningful patterns.
* Establish a Logistic Regression baseline.
* Compare Logistic Regression, Random Forest, and Gradient Boosting.
* Evaluate models using ROC-AUC, PR-AUC, precision, recall, F1-score, and Brier score.
* Examine feature importance and model explanations.
* Discuss limitations, potential bias, and practical risk-management implications.
* Produce a formal 8–12 page PDF research whitepaper.

## Dataset

**Source:** UCI Machine Learning Repository — Default of Credit Card Clients.

Dataset page: https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients

The dataset contains historical credit-card customer information from Taiwan. Its target records whether a customer defaulted on a payment in the following month.

**Target variable:** Default payment next month (binary classification).

The dataset's historical context and geographic limitations will be explicitly considered when interpreting the results.

## Planned Methodology

1. Acquire and validate the dataset.
2. Perform exploratory data analysis and data-quality checks.
3. Define training, validation, and test partitions.
4. Establish a baseline and train candidate models.
5. Compare discrimination, classification performance, and probability calibration.
6. Investigate model explanations and threshold trade-offs.
7. Document findings, limitations, and business implications.
8. Prepare the final research whitepaper.

The final model selection and conclusions will be based on measured results, not assumed in advance.

## Repository Structure

```text
Credit-Risk-Research-Capstone/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   └── results/
├── report/
├── references/
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

The project will document its data source, preprocessing decisions, random seeds, model configurations, evaluation metrics, and software dependencies.

## Limitations

This dataset represents a historical credit-card population in Taiwan. Its results should not be treated as directly generalizable to present-day borrowers or other countries without further validation.

## Research Reference

Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

## Project Status

Initial setup and research design.
