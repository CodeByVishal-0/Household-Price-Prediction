import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.cluster import KMeans

#num_pipeline
num_pipeline=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scale",StandardScaler(with_mean=True))
])

### cat_pipeline
cat_pipeline=Pipeline([
    ("impute",SimpleImputer(strategy='most_frequent')),
    ("onehot",OneHotEncoder())
])

### Ratio Transformation
def ratio(X):
    return X[:,[0]]/X[:,[1]]
def column_name(function_transformer,get_features_in):
    return ["ratio"]

ratio_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("ratio",FunctionTransformer(func=ratio,feature_names_out=column_name)),
    ("standardscaler",StandardScaler(with_mean=True))
])

# cluster similarity

class Similarity4Cluster(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=10, gamma=0.1, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, sample_weight=None, y=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, name=None):
        return [
            f"similarity with {i+1} cluster"
            for i in range(self.n_clusters)
        ]
    
#log transformation
log_pipeline=Pipeline([
    ("simpleimputer",SimpleImputer(strategy="median")),
    ("log",FunctionTransformer(func=np.log,feature_names_out="one-to-one")),
    ("standardscaler",StandardScaler(with_mean=True))
])

# multimodes distribution

def similarity_clms(function_transformer, get_features_in):
    return ["Similarity with Housing age: 35"]


simil = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("similarity",
     FunctionTransformer(
         func=rbf_kernel,
         kw_args=dict(Y=[[35]], gamma=0.1),
         feature_names_out=similarity_clms
         
     )
    ),
    ("standardscaler", StandardScaler())
])

# Final Preprocessing
preprocessings=ColumnTransformer([
    ("bedrooms",ratio_pipeline,['total_bedrooms','total_rooms']),
    ("rooms_per_house",ratio_pipeline,["total_rooms","households"]),
    ("people_per_house",ratio_pipeline,["population","households"]),
    ("log",log_pipeline,["total_bedrooms","total_rooms","population","households","median_income"]),
    ("geo",Similarity4Cluster(),["latitude","longitude"]),
    ("cat",cat_pipeline,["ocean_proximity"]),
    ("simil",simil,['housing_median_age']),
    # ("pass","passthrough",["median_house_value"])
    ("drops","drop",['median_house_value'])
],remainder=num_pipeline)

# def preprocessing():
#     return preprocessings