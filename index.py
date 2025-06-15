import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import median 

bean_data = pd.read_csv("Dry_Bean_Dataset.csv")
# print(bean_data.head())

print(bean_data.shape)
print(bean_data[bean_data.duplicated()])

bean_data = bean_data.drop_duplicates()

print(bean_data.shape)
print(bean_data.isnull().sum())
print(bean_data.info())
print(bean_data.describe().T)
print(bean_data["Class"].value_counts())

# Count Plot of Class type of Bean

plt.figure(figsize=[12,8])
sns.countplot(x="Class", data=bean_data, palette="coolwarm")
plt.show()

# Bar Plot of class and area of bean data

plt.figure(figsize=[12,8])
sns.barplot(x="Class", y="Area", data=bean_data, estimator=median, palette="coolwarm")
plt.show()

# Bar Plot of class and eccentricity of bean data

plt.figure(figsize=[12,8])
sns.barplot(x="Class", y="Eccentricity", data=bean_data, estimator=median, palette="coolwarm")
plt.show()

# Bar Plot of class and Compactness of bean data

plt.figure(figsize=[12,8])
sns.barplot(x="Class", y="Compactness", data=bean_data, estimator=median, palette="coolwarm")
plt.show()

# Hisplot1

f,(ax1, ax2, ax3, ax4, ax5, ax6)=plt.subplots(6, 1, figsize=[12,8])
sns.histplot(x="Area", data=bean_data, kde=True, ax=ax1, palette="coolwarm")
sns.histplot(x="Perimeter", data=bean_data, kde=True, ax=ax2, palette="coolwarm")
sns.histplot(x="MajorAxisLength", data=bean_data, kde=True, ax=ax3, palette="coolwarm")
sns.histplot(x="MinorAxisLength", data=bean_data, kde=True, ax=ax4, palette="coolwarm")
sns.histplot(x="AspectRation", data=bean_data, kde=True, ax=ax5, palette="coolwarm")
sns.histplot(x="Eccentricity", data=bean_data, kde=True, ax=ax6, palette="coolwarm")
plt.tight_layout()
plt.show()


# Hisplot2

f,(ax1, ax2, ax3, ax4, ax5, ax6)=plt.subplots(6, 1, figsize=[12,8])
sns.histplot(x="EquivDiameter", data=bean_data, kde=True, ax=ax1, palette="coolwarm")
sns.histplot(x="Extent", data=bean_data, kde=True, ax=ax2, palette="coolwarm")
sns.histplot(x="Solidity", data=bean_data, kde=True, ax=ax3, palette="coolwarm")
sns.histplot(x="roundness", data=bean_data, kde=True, ax=ax4, palette="coolwarm")
sns.histplot(x="Compactness", data=bean_data, kde=True, ax=ax5, palette="coolwarm")
sns.histplot(x="ShapeFactor1", data=bean_data, kde=True, ax=ax6, palette="coolwarm")
plt.tight_layout()
plt.show()


#PAIRPLOT 

bean_data_num_vars = bean_data[["Area", "Perimeter", "ConvexArea","MajorAxisLength","MinorAxisLength","AspectRation","Eccentricity"]]
plt.figure(figsize=[12,8])
sns.pairplot(bean_data_num_vars)
plt.show()

#Correrelation matrix

corr_mat = bean_data_num_vars.corr()
print(corr_mat)

#HeatMap
figsize=[12,8]
sns.heatmap(corr_mat, annot=True)
plt.show()

#BOX - PLOT
plt.figure(figsize=[12,8])
sns.boxplot(data=bean_data_num_vars, palette="coolwarm")
plt.show()
