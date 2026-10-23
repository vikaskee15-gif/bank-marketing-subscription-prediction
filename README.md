# Bank Marketing Subscription Prediction

**An end-to-end machine-learning project for customer response prediction, campaign targeting, and decision support.**

This project uses the UCI Bank Marketing dataset to estimate which clients are more likely to subscribe to a term deposit. It demonstrates an ML workflow from data inspection and preprocessing through model comparison, validation-based threshold selection, final held-out evaluation, model interpretation, and illustrative campaign economics.

> **Project status:** The notebook is prepared to run from top to bottom and generates its metrics when executed. Saved cell outputs are cleared intentionally; run it before quoting any performance numbers.

## Business problem

A bank has limited outreach capacity and wants to prioritize potential customers. A model can rank customers by estimated subscription likelihood so a campaign team can explore how different targeting thresholds affect precision, recall, and the number of customers selected. Predictions are decision-support signals—not guarantees of subscription.

## Dataset

The notebook downloads `bank-additional-full.csv` from the [UCI Machine Learning Repository — Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing). The file contains 41,188 records with 20 input variables and the binary target `y` (`yes` / `no`). The notebook does not redistribute the dataset.

## Workflow

1. **Load and audit:** Download the source data, check shape, data types, target balance, missing values, and categorical `unknown` values.
2. **Explore:** Review the target distribution and descriptive statistics.
3. **Avoid leakage:** Exclude `duration` because it is known only after the call. Exclude `campaign` for the project's stated pre-contact prioritization scenario.
4. **Split the data:** Use a stratified 60% training, 20% validation, and 20% final test split.
5. **Preprocess safely:** Apply `StandardScaler` to numerical features and `OneHotEncoder(handle_unknown="ignore")` to categorical features inside a scikit-learn `Pipeline` / `ColumnTransformer`.
6. **Compare models:** Evaluate Logistic Regression, Random Forest, and XGBoost on validation data.
7. **Select a threshold:** Choose among the tested probability thresholds using validation F1-score; ties prefer the higher threshold.
8. **Evaluate once on the held-out test set:** Report accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix using the threshold chosen on validation data.
9. **Explain the model:** Inspect XGBoost feature importance and SHAP values.
10. **Translate to decisions:** Explore priority segments, campaign-history associations, subgroup performance, targeting lift, and illustrative economics.

## Evaluation approach

The notebook generates validation metrics for model comparison and final test metrics after the model and threshold have been selected. It reports accuracy alongside precision, recall, F1-score, ROC-AUC, and confusion matrices because the target is imbalanced. No fixed performance numbers are included here: rerun the notebook to produce the results for the current code and environment.

## Business impact — illustrative only

The notebook demonstrates campaign calculations using placeholder contact-cost and subscription-value assumptions. These are **not verified profit or causal incremental impact**. Historical labels alone cannot establish which customers subscribed because of outreach versus which would have subscribed anyway. Replace the placeholder values with actual campaign costs and contribution margin/customer lifetime value, and use a randomized control group or another credible uplift design to estimate incremental impact.

## Limitations and next steps

- **Temporal generalization:** The dataset is historical and ordered by time. A random stratified split is a useful baseline, but a chronological/out-of-time evaluation should be added before making claims about future campaigns.
- **Context transfer:** Patterns may not generalize to another bank, time period, product, or customer population.
- **Threshold choice:** The best operating point depends on campaign capacity and the relative cost of false positives and false negatives. F1-score is only one possible objective.
- **Calibration:** Evaluate probability calibration before interpreting scores as accurate probabilities.
- **Subgroups:** Group sizes differ; inspect uncertainty and fairness before operational use.
- **Causality:** Model-based targeting versus an expected random baseline is descriptive and does not substitute for an experiment.

## Run the project

### Recommended: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `bank_marketing_subscription_prediction.ipynb`.
3. Select **Runtime → Run all**.
4. Let the notebook download the dataset and install XGBoost/SHAP when prompted by the notebook cells.
5. Review the fresh validation comparison, selected threshold, held-out test metrics, SHAP plot, and business-analysis tables.

### Local environment

Python with the packages listed in [`requirements.txt`](requirements.txt) is needed. Open the notebook in Jupyter and run the cells from top to bottom. The notebook downloads the dataset directly from UCI.

## Repository structure

```text
bank-marketing-subscription-prediction/
├── README.md
├── bank_marketing_subscription_prediction.ipynb
├── requirements.txt
└── .gitignore
```

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost, SHAP, Jupyter/Google Colab.

## Source and attribution

Dataset citation: Moro, S., Rita, P., & Cortez, P. (2014). [Bank Marketing dataset](https://doi.org/10.24432/C5K306). UCI Machine Learning Repository. The dataset is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) licence](https://creativecommons.org/licenses/by/4.0/). The related paper is [A Data-Driven Approach to Predict the Success of Bank Telemarketing](https://doi.org/10.1016/j.dss.2014.03.001), *Decision Support Systems*, 62, 22–31.
