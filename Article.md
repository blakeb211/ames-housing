This article outlines a cleaner modeling process using Sklearn and a few companion libraries. It guides the reader on how to create the full range of ML models--regularized linear, bagged and boosted trees, to deep neural networks and automated modeling (automl)--on the Ames housing dataset.

A very small survey of working data scientists suggests that experienced and inexperienced Sklearn users will find something new here.

The code is here. Steps left out of the article can be found there easily. 

Companion libraries help us reduce our boilerplate and add new functionality to the widely used Sklearn.
* Feature-Engine - provides simpler tranformers, reducing boilerplate and tightening up pipelines
* Yellowbrick - provides wrappers over estimators to make common visualizations quickly
* XGBoost - provides an additional (and widely popular) gradient boosting implementation
* AutoSklearn - provides AutoML functionality

After ingestion and creating a train-test split, we construct preprocessing recipes called **Pipelines** to automate our preprocessing steps such as scaling and one-hot encoding. These produce clean code and guard against subtle data leakage during K-Fold validation.

**There is an ingestion script** so that your exploration and modeling files aren't littered with copy-pasted code. The modeling and exploration notebooks or scripts stay independent from one another and changes to ingestion / cleaning can be made once.

**Write tighter pipeline code with SklearnTransformWrapper and make_pipeline**
Feature Engine's *SklearnTransformWrapper* wraps our StandardScaler() and OneHotEncoder() so that we can put them directly into our Pipeline object. Sklearn's *make_pipeline* names the steps for us. GridSearchCV tunes the hyperparameters through the name of each step. No more ColumnTransformer and no more manual pipeline construction. 

```
```

**Yellowbrick has helper classes and functions to automate creation of various important graphics and keep our thinking (mostly) up above the matplotlib api.**
Your hyperparameter search with GridSearchCV may have covered multiple parameters. Choose one and validate it with Yellowbrick's *ValidationCurve* to make sure things look OK. 

```
```

Sklearn has estimators for everything from regularized regression, to support vector machine, single decision trees, bagged versions of trees, boosted trees, and deep neural nets. Yet XGBoost is widely touted for speed and range of parameters, and AutoML is a hot topic. **Companion libaries XGBoost and AutoSklearn complete the range of models (estimators) already present in Sklearn** by adding these functionalities to your modeling project. 

```
```









