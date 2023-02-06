# From Least Squares to AutoML 

# What 
- Demonstate machine learning techniques and concepts on the Ames housing dataset. 

# Why
- Demonstrate knowledge obtained from the resources at the bottom on a popular dataset.

# Plan
1. Ingestion
1. Exploration
1. Feature Engineering
1. Model (tuning, validation)
1. Miscellaneous concepts: regularization, PCA, metrics, target variable transformation 
1. Feature Importance Analysis
1. Write algorithm overviews
1. Check impact of additional feature engineering on top two models 
1. Clean up comments and repo presentation
1. Write an article

# Models
* Linear, KNN, Decision Tree, Bagged Trees, Random Forest, Gradient Boosted, AutoML 

# Algorithm overviews
* Linear (Ordinary Least Squares)
* K-Nearest Neighbor
* Decision Tree
* Bagged Trees
* Random Forest
* Gradient Boosted Trees
* AutoML 

# Feature Engineering Notes 
1. Imputed missing values using 'most frequent value' and k-nearest neighbor.
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
1. Hands-on Machine Learning with R book, Boehmke and Greenwell
1. Vectors Matrices and Least Squares book, Boyd and Vandenberghe
1. API docs and code for the various libraries Sklearn, Xgboost, AutoSklearn, Numpy, Pandas 

# Python Resources
1. ThinkPython
1. Automate The Boring Stuff
1. ThinkStats