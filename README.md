# Explainable Credit Risk Prediction

**A Data Science Research Capstone on Credit Default Prediction, Model Explainability, Calibration, and Fairness Assessment**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Research%20Completed-success)]()

## Project Overview

This research project investigates machine-learning approaches for predicting credit-card default using historical customer data. It evaluates predictive performance, probability quality, feature-level explanations, decision thresholds, and differences in model performance across demographic subgroups.

The project follows an end-to-end Data Science workflow, from dataset validation and exploratory analysis to model comparison, explainability, uncertainty analysis, and research documentation.

The objective is not simply to maximize predictive performance, but to understand the trade-offs and limitations involved in developing credit-risk prediction models.

## Research Question

How effectively can machine-learning models predict credit-card default, and how do their predictive performance, probability quality, explainability, and subgroup error rates compare?

## Key Results

The final evaluation compared Logistic Regression, Random Forest, and Gradient Boosting on the same final test split.

| Model                 |    ROC-AUC | Average Precision | Brier Score |   Log Loss |
| --------------------- | ---------: | ----------------: | ----------: | ---------: |
| Logistic Regression   |     0.7376 |            0.5142 |      0.1432 |     0.4603 |
| Random Forest         |     0.7781 |            0.5465 |      0.1393 |     0.4401 |
| **Gradient Boosting** | **0.7849** |        **0.5517** |  **0.1342** | **0.4287** |

Gradient Boosting achieved the strongest ROC-AUC and average precision among the evaluated models, along with the lowest Brier score and log loss.

These findings apply to this dataset and evaluation procedure; they do not establish performance on other populations.

## Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default)

* **Observations:** 30,000
* **Columns:** 24, including the target
* **Input features:** 23
* **Target:** `default_payment_next_month`
* **Observed default rate:** 22.12%
* **Missing values:** 0
* **Exact duplicate rows:** 35, retained in the working dataset

The dataset represents a historical credit-card customer population in Taiwan. The working CSV does not contain the original customer ID, so identical rows cannot automatically be assumed to represent the same individual.

## Methodology

The project covers the following stages:

1. **Data validation:** Check dataset dimensions, missing values, duplicate rows, target values, and data quality.
2. **Exploratory data analysis:** Examine default distribution, age, credit limits, and repayment-status patterns.
3. **Model development:** Establish a Logistic Regression baseline and compare it with Random Forest and Gradient Boosting.
4. **Model evaluation:** Assess ROC-AUC, average precision, accuracy, precision, recall, F1-score, Brier score, and log loss.
5. **Probability assessment:** Examine calibration and the quality of predicted probabilities.
6. **Explainability:** Use SHAP to investigate influential features and model behaviour.
7. **Threshold analysis:** Compare classification outcomes at different probability thresholds.
8. **Subgroup evaluation:** Examine recall and false-positive rates across sex-coded categories and age groups, including Wilson 95% confidence intervals.
9. **Research documentation:** Consolidate the methodology, results, limitations, and references into a PDF report.

### Evaluation Design

The final evaluation uses a stratified 60% training, 20% validation, and 20% test split with random seed `2026`. Candidate model selection uses validation ROC-AUC, followed by evaluation on the final test split.

**Important qualification:** Earlier exploratory modelling used the same dataset before the final split was introduced. The final test set is therefore not an entirely untouched external evaluation.

## Explainability

SHAP analysis identified the following leading features by mean absolute SHAP value:

| Feature     | Mean Absolute SHAP |
| ----------- | -----------------: |
| `PAY_0`     |             0.4715 |
| `LIMIT_BAL` |             0.1459 |
| `PAY_3`     |             0.0818 |
| `PAY_AMT3`  |             0.0809 |
| `PAY_AMT2`  |             0.0750 |

The results indicate that repayment-status variables strongly influenced the model's predictions in this dataset.

SHAP importance describes model behaviour and associations. It does not establish causality or demonstrate that the model is free from unfair effects.

## Decision Threshold Analysis

The threshold comparison illustrates the trade-off between identifying defaults and generating false-positive predictions.

| Metric          | Threshold 0.25 | Threshold 0.50 |
| --------------- | -------------: | -------------: |
| Accuracy        |         0.7920 |         0.8213 |
| Precision       |         0.5265 |         0.6798 |
| Recall          |         0.5923 |         0.3632 |
| F1-score        |         0.5574 |         0.4735 |
| True positives  |            786 |            482 |
| False positives |            707 |            227 |

The 0.25 threshold achieved higher recall and F1-score, while the 0.50 threshold achieved higher accuracy and precision. The appropriate threshold depends on the costs of missed defaults and false alarms.

The threshold of 0.25 was selected using validation-set F1 among the tested thresholds; it is not a universal lending policy.

## Subgroup Evaluation and Uncertainty

The project evaluates recall and false-positive rates across sex-coded categories and age groups, with Wilson 95% confidence intervals.

The estimates vary between groups, and uncertainty is particularly wide for the age-60-and-over test subgroup, which contains only 60 records.

These differences warrant further investigation but do not independently establish discrimination or fairness. A complete fairness assessment would require additional analysis and context.

## Repository Structure

```text
Credit-Risk-Research-Capstone/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── explore_dataset.py
│   ├── validate_dataset.py
│   ├── train_baseline.py
│   ├── compare_models.py
│   ├── evaluate_calibration.py
│   ├── explain_model.py
│   ├── fairness_analysis.py
│   ├── fairness_uncertainty_analysis.py
│   ├── final_evaluation.py
│   ├── threshold_analysis.py
│   ├── final_threshold_evaluation.py
│   └── generate_research_report.py
├── outputs/
│   ├── figures/
│   └── results/
├── report/
│   └── credit_risk_research_report_revised.pdf
├── references/
├── requirements.txt
├── .gitignore
└── README.md
```

## Research Report

The final research report consolidates the dataset validation, methodology, model comparison, SHAP analysis, threshold evaluation, subgroup uncertainty analysis, limitations, and references.

**Report:** [`report/credit_risk_research_report_revised.pdf`](report/credit_risk_research_report_revised.pdf)

## Reproducibility

The repository includes Python source scripts, saved experiment results, figures, dependency specifications, and the report-generation script.

To set up the environment on Windows PowerShell:

```powershell
git clone https://github.com/Debankumardas/Credit-Risk-Research-Capstone.git
cd Credit-Risk-Research-Capstone

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

To regenerate the PDF report after the required analysis outputs are available:

```powershell
python .\src\generate_research_report.py
```

## Limitations and Responsible Use

* **Population generalization:** The dataset is historical and specific to Taiwan. Results should not be assumed to generalize to present-day borrowers or other countries.
* **Evaluation independence:** Earlier exploratory analysis used the same dataset; independent external validation remains necessary.
* **Threshold selection:** Operating thresholds depend on the relative consequences of false positives and false negatives.
* **Fairness:** Subgroup confidence intervals are informative but do not constitute a complete fairness assessment.
* **Explainability:** SHAP values describe model behaviour, not causal relationships.
* **Operational readiness:** This is an educational research project, not a validated or approved system for making real credit decisions.

## References

1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
2. University of California, Irvine. *Default of Credit Card Clients*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350/default
3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems, 30*.
4. Scikit-learn developers. *Scikit-learn documentation*. https://scikit-learn.org/stable/

---

**Project Status:** Research analysis and report generation completed.

**Author:** [Deban Kumar Das D](https://github.com/Debankumardas)
