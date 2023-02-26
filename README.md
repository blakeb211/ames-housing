# From Linear Models To AutoML with Sklearn & Friends

# What 
- Demonstration notebooks showing the full modeling pipeline for linear models to neural nets to trees to AutoML 
with common python libaries, Sklearn, and its companion libraries.

# Why
- Apply and explain knowledge from the resources at the bottom to a popular dataset.

# How
- *ingestion.py* get called by each modeling notebook to generate dataframes for modeling
- *metadata_helper.py* gets called by each modeling notebook to save the modeling metrics to csv and html files
- Each modeling notebook (prefixed with *model_*) is independent from the others. They only depend on the two ingestion and score-saving scripts above.

# Recommended setup
- Use `virtualenv` to create a virtual environment with Python 3.9. Activate the environment. Install the dependencies with pip.
``` 
python3.9 -m venv env 
source .env/bin/activate 
pip install -r requirements.txt
```

# Plan
1. Ingestion and Wrangling
1. Exploration
1. Model (tuning, validation)
1. Clean up comments
1. Generate scores figure
1. Save any other necessary figures
1. Make and test code examples in article
1. Finish article
1. Publish to Medium
1. Submit to TDS

## Models
* Linear, KNN, Decision Tree, Bagged Trees, Random Forest, Gradient Boosted, AutoML 

## Simplified Algorithm Overviews
### Regularized Linear Regression (Ordinary Least Squares)  
Solve the least squares problem to get weights (slopes) of each feature. Feature scaling is important. Regularization reduces co-linearity.

### K-Nearest Neighbor
Predict the target value as the average of a certain number of nearest neighbors. Nearest neighbor determined from a distance metric, e.g. the Euclidiean norm (square root of summed squares of differences), or the manhattan distance (sum of absolute values of differences). Feature scaling is important.

### Decision Tree
Cycle through features to find the one that best splits the target variable into two groups. Add split (branch) to the tree. Repeat. A distance metric is used to measure how good each split is.

### Bagged Trees
A group of single trees. Sample with replacement from the training data (bootstrap sampling). Train decision tree on this bootstrap sample. Repeat. Combine results of all trees to get the model prediction.

### Random Forest
Bagged trees with additional randomness in the feature cycling step. Instead of cycling through all features for best split, only cycle through a certain percentage of them. Both bagging and feature randomization can add a large improvment to a single decision tree, despite what intuition might tell us. 

### Gradient Boosted Trees
One relatively shallow tree is created. The model errors from that tree are treated as training data for a new tree. Repeat. Popular packages like XGBoost add in several other techniques like bagging, feature randomization, regularization, and more. 

### Neural Network
The layer sizes of the neural network (layer sizes) is chosen. Since the feedforward NN used here has 4 layers, it is a "deep" net. Numerical optimization (backpropagation, gradient descent) is done to find the weights that minimize the performance metric (loss function). See the GPT transformer diagram for an idea of how complex "deep learning" network architectures can get.

### AutoML  
Algorithms for feature pre-processing are automatically applied, partially informed by a database of what worked on previous similar datasets. Several models are created and ranked. A group (ensemble) of models with different error shapes are stacked together. A voting procedure combines their answers to give the final prediction. 

# Feature Engineering Notes 
1. Imputed missing values using 'most frequent value' and k-nearest neighbor.
1. Removed 5 outliers observed in scatter plot of GrLivArea and SalePrice. 

# Miscellaneous Observations
1. Validation Curves shapes are model dependent. 
1. Training time for SVR models with an rbf kernel depends heavily on the gamma parameter
1. For a single decision tree, the validation curve may select a more complicated model 
than what will generalize best to new data. This did not occur (for this dataset) with tree ensemble algorithms.
1. Bagging (bootstrap aggregation) improves over a single decision tree quite a bit

# Machine Learning Resources
1. Georgetown Data Science Certificate Program
1. Hands-on Machine Learning with R book, Boehmke and Greenwell
1. Vectors Matrices and Least Squares book, Boyd and Vandenberghe
1. ThinkStats
1. API docs and code for the various libraries Sklearn, Xgboost, AutoSklearn, Numpy, Pandas
