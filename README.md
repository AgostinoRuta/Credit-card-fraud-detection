## Credit Card Fraud Detection

This notebook focuses on classifying credit card transactions to identify fraudulent ones, using the known dataset available on Kaggle 
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This binary classification problem is characterize by an highly unbalanced 
dataset since out of 284,807 transactions, only 492 are fraudulent. Following an exploratory data analysis (EDA), the dataset is split 
into training and testing sets. The final model chosen will consist of a multiple ensemble of bagged algorithms. The out-of-sample 
predictions of the trained model yield the following statistics:

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**         | 0.999631  |
| **Balanced Accuracy**| 0.923417  |
| **ROC AUC**          | 0.983186  |
| **Log Loss**         | 0.002136  |
| **Precision**        | 0.932584  |
| **Recall**           | 0.846939  |
| **F1 Score**         | 0.887701  |
| **Kappa**            | 0.887516  |
| **Average Precision**| 0.790105  |
