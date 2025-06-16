import sklearn
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

bean_data = pd.read_csv("Dry_Bean_Dataset.csv")
# print(bean_data.head())

x = bean_data.drop(columns=["Class"])
y = bean_data["Class"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# print(x_train)
# print(y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# print(x_train_scaled)

x_train_scaled_df = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
print(x_train_scaled_df.head())
