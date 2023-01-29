"""
Cleaning functions for the Ames housing dataset.


#### RECIPE #####
# Encode ordered categorical features e.g. Qual,Cond,QC
# Impute missing values in numeric and nominal columns
# Remove near-zero variance nominal variables e.g. Street
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer


def make_raw():
    """
    Reads in the raw dataset downloaded from Kaggle
    """
    df = pd.read_csv("./data/AmesHousing.xls").rename(columns=lambda x: x.replace(" ", ""))
    return df

def make_cleaned():
    """
    Returns a cleaned dataframe with no missing values. One-hot encoding and scaling are left for modeling phase.
    """
    raw = make_raw()
    cleaning = raw.copy()

    # Convert ordered categorical values to numbers
    ordinal_vals = ("Po", "Fa", "TA", "Gd", "Ex")

    ordinal_cols = list(set(
        raw.filter(regex="Qual$").columns.to_list()
        + raw.filter(regex="QC$").columns.to_list()
        + raw.filter(regex="Qu$").columns.to_list()
        + raw.filter(regex="Cond$").columns.to_list()
    ) - set(["OverallQual", "OverallCond"]))

    ordinal_imputer = SklearnTransformerWrapper(transformer=(OrdinalEncoder(categories=[ordinal_vals] * len(
        ordinal_cols), handle_unknown="use_encoded_value", unknown_value=np.nan)))

    cleaning.loc[:, ordinal_cols] = pd.DataFrame(
        ordinal_imputer.fit_transform(make_raw()[ordinal_cols]))

    # Impute nans
    most_freq_imputer = SklearnTransformerWrapper(
        transformer=(SimpleImputer(strategy="most_frequent")))

    numeric_imputer = SklearnTransformerWrapper(
        transformer=KNNImputer(n_neighbors=3, missing_values=np.nan))

    # Impute numerics using KNN
    numerics = cleaning.select_dtypes(np.number).columns.tolist()
    cleaning.loc[:, numerics] = numeric_imputer.fit_transform(
        cleaning.select_dtypes(np.number))

    # Impute all remaining using most frequent
    # @Note that there is likely room to improve here if we utilize missing data trends in Bsmt variables, 
    # missing Alley as None and things like that.
    cleaning = most_freq_imputer.fit_transform(cleaning)

    # Convert dtypes because SimpleImputer made everything an object
    cleaning = cleaning.convert_dtypes(convert_string=False)

    # Remove near zero variance columns
    nzv_remover = SklearnTransformerWrapper(
        transformer=VarianceThreshold(threshold=0.1))
    cleaning = nzv_remover.fit_transform(cleaning)

    assert sum(cleaning.isna().sum()) == 0

    return cleaning
