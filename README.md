# House Prices Project - Ames Dataset

# What 
- Demonstate machine learning techniques and concepts on the Ames housing dataset. 

# Plan
1. Ingestion
1. Exploration
1. Feature Engineering
1. Model (tuning, validation)
1. Feature Importance Analysis
1. Miscellaneous topics: regularization, PCA, metrics, target variable transformation 

# Models
* KNN, Linear, Decision Tree, Bagged Trees, Random Forest, Gradient Boosted, Stacked

# Feature Engineering Notes 
1. Removed 5 outliers observed in scatter plot of GrLivArea and SalePrice. 

## Observations
1. l1 norm is the cityblock or manhattan distance. l2 norm is the euclidean or minkowski distance.
1. Validation Curves shapes are model dependent. 
1. Training time for SVR models with an rbf kernel depends heavily on the gamma parameter
1. For a single decision tree, the validation curve may select a more complicated model 
than what will generalize best to new data. This did not occur (for this dataset) with tree ensemble algorithms.
1. Bagging (bootstrap aggregation) improves over a single decision tree quite a bit

# Resources
1. Georgetown Data Science Certificate Program
1. Hands-on Machine Learning with R book
1. Vectors Matrices and Least Squares (VMLS) book
1. Sklearn and Xgboost api docs