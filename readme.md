# Dry Bean Classification Project

A comprehensive machine learning project for classifying dry bean seeds based on their morphological features. This repository covers the full workflow: data exploration, feature engineering, model training, and evaluation.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering & Selection](#feature-engineering--selection)
- [Model Training & Evaluation](#model-training--evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)
- [Author & License](#author--license)

---

## Dataset

The dataset [`Dry_Bean_Dataset.csv`](Dry_Bean_Dataset.csv) contains morphological measurements for dry bean samples. Each row represents a sample with the following columns:

- **Area**
- **Perimeter**
- **MajorAxisLength**
- **MinorAxisLength**
- **AspectRation**
- **Eccentricity**
- **ConvexArea**
- **EquivDiameter**
- **Extent**
- **Solidity**
- **Roundness**
- **Compactness**
- **ShapeFactor1**
- **ShapeFactor2**
- **ShapeFactor3**
- **ShapeFactor4**
- **Class** (target variable)

---

## Project Structure

```
├── Dry_Bean_Dataset.csv      # Dataset file
├── index.py                  # EDA and visualization script
├── pipleline_feature.py      # Feature selection and ML pipeline script
└── readme.md                 # Project documentation
```

---

## Requirements

Install the required Python packages:

```sh
pip install pandas numpy seaborn matplotlib scikit-learn imblearn
```

---

## Exploratory Data Analysis (EDA)

Performed in `index.py`:

- **Data Cleaning:**  
    - Checks for duplicates and missing values  
    - Removes duplicate rows

- **Descriptive Statistics:**  
    - Displays dataset shape, info, and summary statistics  
    - Shows class distribution

- **Visualizations:**  
    - Count plots for class distribution  
    - Bar plots and histograms for feature distributions  
    - Pairplots for feature relationships  
    - Correlation matrix and heatmap  
    - Boxplots for feature spread and outliers

---

## Feature Engineering & Selection

Implemented in `pipleline_feature.py`:

- **Data Splitting:**  
    - Stratified train-test split

- **Scaling:**  
    - Standardizes features with `StandardScaler`

- **Feature Selection:**  
    - Uses `SelectKBest` (ANOVA F-value) to select top features  
    - Evaluates model performance for different feature counts  
    - Visualizes feature scores

---

## Model Training & Evaluation

- **Model:**  
    - Logistic Regression

- **Metrics:**  
    - Accuracy  
    - Precision (macro)  
    - Recall (macro)  
    - F1 Score (macro)  
    - Classification report  
    - Confusion matrix

- **Feature Selection Impact:**  
    - Assesses model performance with varying feature subsets  
    - Visualizes feature importance

---

## How to Run

1. Clone or download the repository.
2. Install dependencies.
3. Place `Dry_Bean_Dataset.csv` in the project directory.

**Run EDA:**
```sh
python index.py
```
Outputs data statistics and visualizations.

**Run Feature Selection & Model Pipeline:**
```sh
python pipleline_feature.py
```
Outputs selected features, feature scores, and model evaluation metrics.

---

## Results

- EDA reveals feature distributions, relationships, and class balance.
- Feature selection identifies the most informative features.
- Logistic regression achieves solid performance; further improvements possible with advanced models or tuning.

---

## References

- [UCI Machine Learning Repository: Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- [scikit-learn documentation](https://scikit-learn.org/)
- [pandas documentation](https://pandas.pydata.org/)
- [seaborn documentation](https://seaborn.pydata.org/)

---

## Author & License

Developed by [Raman Kumar] for educational purposes.

Feel free to customize the author and license sections as needed.
