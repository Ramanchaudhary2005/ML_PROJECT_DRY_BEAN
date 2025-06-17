import sklearn
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
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
# print(x_train_scaled_df.head())

model = LogisticRegression(max_iter=10000)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
# print(y_pred)

predict_vs_actual_df = pd.DataFrame({"Predicted": y_pred, "Actual": y_test})
# print(predict_vs_actual_df)

conf_mat_logreg = pd.crosstab(predict_vs_actual_df["Predicted"], predict_vs_actual_df["Actual"])
# print(conf_mat_logreg)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
fiscore = f1_score(y_test, y_pred, average="macro")

# print(accuracy)
# print(precision)
# print(recall)
# print(fiscore)

classification_rep = classification_report(y_test, y_pred)
# print(classification_rep, sep="\n")

pipeline = Pipeline(steps=[("Scaler", scaler),("model", model)])
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
print(y_pred)

# FETAURE SELACTION

feature_selection = SelectKBest(score_func=f_classif, k=6)

feature_selection.fit(x_train, y_train)
feature_selection.get_feature_names_out()

feature_selection.scores_

f1_scr, acc_src, pre_scr, rec_scr = ([] for lst in range(0,4))
for i in range(1, len(x.columns)+1):
    feature_selection = SelectKBest(score_func=f_classif, k=i)
    pipeline = Pipeline(steps=[("Scaler", scaler), ("feature_selection", feature_selection), ("model", model)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    # print("Accuracy Score (k={}): {}".format(i, accuracy_score(y_test, y_pred)))
    # print("Precision Score (k={}): {}".format(i, precision_score(y_test, y_pred, average="macro")))
    # print("Recall Score (k={}): {}".format(i, recall_score(y_test, y_pred, average="macro")))
    # print("F1 Score (k={}): {}".format(i, f1_score(y_test, y_pred, average="macro")))


mask = pipeline["feature_selection"].get_support()
print(mask)

selected_features = x_train.columns[mask]
print("Selected Features:", selected_features)

# Visualizing the selected features
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_features, y=pipeline["feature_selection"].scores_[mask])
plt.title("Selected Features Scores")
plt.xlabel("Features")
plt.ylabel("Scores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

selected_features_df = pd.DataFrame({"Features": selected_features, "Scores": pipeline["feature_selection"].scores_[mask]})
selected_features_df.sort_values(by="Scores", ascending=False, inplace=True)
print(selected_features_df)