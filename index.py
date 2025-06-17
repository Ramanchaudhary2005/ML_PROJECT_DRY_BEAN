import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import median

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load data
bean_data = pd.read_csv("Dry_Bean_Dataset.csv")

# Data overview
print("Initial shape:", bean_data.shape)
print("Duplicated rows:\n", bean_data[bean_data.duplicated()])
bean_data = bean_data.drop_duplicates()
print("Shape after removing duplicates:", bean_data.shape)
print("Missing values:\n", bean_data.isnull().sum())
print(bean_data.info())
print(bean_data.describe().T)
print("Class distribution:\n", bean_data["Class"].value_counts())

# Count Plot of Class type of Bean
plt.figure(figsize=[12, 8])
sns.countplot(x="Class", data=bean_data, palette="coolwarm")
plt.title("Count of Each Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
save_plot("count_of_each_bean_class.png")

# Median Area by Class
plt.figure(figsize=[12, 8])
sns.barplot(x="Class", y="Area", data=bean_data, estimator=median, palette="coolwarm")
plt.title("Median Area by Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Median Area")
plt.xticks(rotation=45)
save_plot("median_area_by_class.png")

# Median Eccentricity
plt.figure(figsize=[12, 8])
sns.barplot(x="Class", y="Eccentricity", data=bean_data, estimator=median, palette="coolwarm")
plt.title("Median Eccentricity by Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Median Eccentricity")
plt.xticks(rotation=45)
save_plot("median_eccentricity_by_class.png")

# Median Compactness
plt.figure(figsize=[12, 8])
sns.barplot(x="Class", y="Compactness", data=bean_data, estimator=median, palette="coolwarm")
plt.title("Median Compactness by Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Median Compactness")
plt.xticks(rotation=45)
save_plot("median_compactness_by_class.png")

# Distribution Plots
features1 = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity"]
fig, axes = plt.subplots(len(features1), 1, figsize=[12, 14])
for i, feature in enumerate(features1):
    sns.histplot(x=feature, data=bean_data, kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
save_plot("distribution_features1.png")

features2 = ["EquivDiameter", "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor1"]
fig, axes = plt.subplots(len(features2), 1, figsize=[12, 14])
for i, feature in enumerate(features2):
    sns.histplot(x=feature, data=bean_data, kde=True, ax=axes[i], color="coral")
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
save_plot("distribution_features2.png")

# Pairplot
num_vars = ["Area", "Perimeter", "ConvexArea", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity"]
sns.pairplot(bean_data[num_vars + ["Class"]], hue="Class", palette="coolwarm", diag_kind="kde")
plt.suptitle("Pairplot of Numerical Features by Bean Class", y=1.02)
plt.savefig("pairplot_by_class.png")
plt.close()

# Correlation Heatmap
corr_mat = bean_data[num_vars].corr()
plt.figure(figsize=[10, 8])
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
save_plot("correlation_matrix.png")

# Boxplot
plt.figure(figsize=[14, 8])
sns.boxplot(data=bean_data[num_vars], palette="coolwarm")
plt.title("Boxplot of Numerical Features")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.xticks(rotation=45)
save_plot("boxplot_numerical_features.png")

# Violin Plot
plt.figure(figsize=[12, 8])
sns.violinplot(x="Class", y="Area", data=bean_data, palette="coolwarm")
plt.title("Distribution of Area by Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Area")
plt.xticks(rotation=45)
save_plot("violinplot_area_by_class.png")

# Swarm Plot
plt.figure(figsize=[12, 8])
sns.swarmplot(x="Class", y="Compactness", data=bean_data, palette="coolwarm", size=2)
plt.title("Compactness Distribution by Bean Class")
plt.xlabel("Bean Class")
plt.ylabel("Compactness")
plt.xticks(rotation=45)
save_plot("swarmplot_compactness_by_class.png")
