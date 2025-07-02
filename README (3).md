# Light Curve Feature Extraction and Classification

This repository demonstrates a workflow for astronomical light curve data analysis, including data preparation, feature extraction, supervised classification, and dimensionality reduction using PCA. The code is written in Python and is suitable for variable star or transient detection datasets.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Data Loading and Exploration](#1-data-loading-and-exploration)
  - [2. Light Curve Visualization](#2-light-curve-visualization)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Building the Feature DataFrame](#4-building-the-feature-dataframe)
  - [5. Supervised Classification](#5-supervised-classification)
  - [6. Prediction Example](#6-prediction-example)
  - [7. Classification with All Features](#7-classification-with-all-features)
  - [8. Dimensionality Reduction using PCA](#8-dimensionality-reduction-using-pca)
  - [9. Classification with PCA Data](#9-classification-with-pca-data)
- [Notes](#notes)

## Overview

- **Loads** astronomical light curve data and metadata from CSV files.
- **Visualizes** light curves for individual objects.
- **Extracts features** from light curves using [`feets`](https://feets.readthedocs.io/).
- **Performs supervised classification** using logistic regression.
- **Applies PCA** for dimensionality reduction.
- **Evaluates** classification performance with F1 score and accuracy.

## Requirements

- Python 3.x
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [feets](https://feets.readthedocs.io/) (`pip install feets`)

## Usage

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
raw_data = pd.read_csv("training_set.csv")
metadata_data = pd.read_csv("training_set_metadata.csv")

# Display the first few rows
display(raw_data.head())
display(metadata_data.head())

# Merge data with metadata
df = raw_data.merge(metadata_data, on="object_id")
display(df.head())

# Data shape and unique targets
print("Shape of the data:", df.shape)
print("Unique values in target column:", df.target.unique())
print("Shape of target column:", df.target.unique().shape)
```

### 2. Light Curve Visualization

```python
# Filter for one object
one_light_curve = df[df["object_id"] == 615]
display(one_light_curve.head())
display(one_light_curve.object_id.unique())

# Plot the light curve
fig, ax = plt.subplots()
ax.scatter(one_light_curve["mjd"], one_light_curve["flux"], label="Flux")
ax.set_xlabel("MJD")
ax.set_ylabel("Flux")
ax.set_title("Light Curve for object_id=615")
ax.legend()
plt.show()
```

### 3. Feature Extraction

```python
import feets

# Prepare light curve table
light_curve = df[["object_id", "mjd", "passband", "flux", "flux_err", "target"]]
display(light_curve.head())
print("Shape of the data frame: ", light_curve.shape)

# Select features (ensure all are supported by your feets version)
feature_names = [
    "Amplitude",
    "AndersonDarling",
    "MaxSlope",
    "Mean",
    "Meanvariance",
    "MedianAbsDev",
    "Rcs",
    "Skew",
    "Std"
]
print("Feature names:", feature_names)

# Compute features for each object and passband
object_ids = light_curve["object_id"].unique()
passband = 0
feature_values = np.zeros((len(object_ids), len(feature_names)))
targets = np.zeros(len(object_ids))

for index, object_id in enumerate(object_ids):
    light_curve_object = light_curve[
        (light_curve["object_id"] == object_id) & (light_curve["passband"] == passband)
    ]
    lc_data = light_curve_object[["mjd", "flux", "flux_err"]]
    if lc_data.shape[0] < 2:
        print(f"Skipping object_id {object_id}: not enough data points in passband {passband}")
        continue
    time = lc_data["mjd"].values
    magnitude = lc_data["flux"].values
    error = lc_data["flux_err"].values
    fs = feets.FeatureSpace(only=feature_names, data=["time", "magnitude", "error"])
    try:
        features, values = fs.extract(time, magnitude, error)
        feature_values[index, :] = values
        targets[index] = light_curve_object.iloc[0, -1]
    except Exception as e:
        print(f"Feature extraction failed for object_id {object_id}: {e}")
        continue

print("Feature values shape:", feature_values.shape)
print("Targets shape:", targets.shape)
```

### 4. Building the Feature DataFrame

```python
# Concatenate features and targets
features_values_with_targets = np.c_[object_ids, feature_values, targets]
print("Shape of the features values with targets:", features_values_with_targets.shape)

# Column names
column_names = ["object_id"] + feature_names + ["target"]
print("Column names:", column_names)

# Build DataFrame
features_light_curves = pd.DataFrame(features_values_with_targets, columns=column_names)
display(features_light_curves.head())
```

### 5. Supervised Classification

```python
# Use only two features for demonstration
x = features_light_curves.loc[:, ["Amplitude", "AndersonDarling"]]
y = features_light_curves.loc[:, ["target"]]
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logistics_regression_model = LogisticRegression(multi_class="auto")
logistics_regression_model.fit(x, y)

print("f1 score:", f1_score(y, logistics_regression_model.predict(x), average="weighted"))
print("Accuracy:", logistics_regression_model.score(x, y))
```

### 6. Prediction Example

```python
# Example prediction
variable_1 = 117
variable_2 = 1.00
variables = np.array([[variable_1, variable_2]])
prediction = logistics_regression_model.predict(variables)
print("Prediction for variables", variables, ":", prediction[0])
```

### 7. Classification with All Features

```python
W = features_light_curves.loc[:, feature_names]
Y = features_light_curves.loc[:, ["target"]]
print("Shape of W:", W.shape)
print("Shape of Y:", Y.shape)

logistics_regression_model_all = LogisticRegression(
    max_iter=10000,
    multi_class="auto",
    class_weight="balanced",
    solver="lbfgs",
    random_state=0
)
logistics_regression_model_all.fit(W, Y)
print("f1 score: ", f1_score(Y, logistics_regression_model_all.predict(W), average="weighted"))
print("Accuracy:", logistics_regression_model_all.score(W, Y))
```

### 8. Dimensionality Reduction using PCA

```python
from sklearn.decomposition import PCA

A = features_light_curves[feature_names]
n_components = 3
pca_model = PCA(n_components=n_components)
principal_componenets = pca_model.fit_transform(A)

principal_components_df = pd.DataFrame(
    data=principal_componenets,
    columns=[f"Principal Component {i+1}" for i in range(n_components)]
)
principal_components_df["object_id"] = features_light_curves["object_id"].values
principal_components_df["target"] = features_light_curves["target"].values
display(principal_components_df.head())

# Plot explained variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(np.arange(n_components), pca_model.explained_variance_ratio_, alpha=0.7, color='blue')
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Explained Variance")
plt.show()
```

### 9. Classification with PCA Data

```python
logistics_regression_pca = LogisticRegression(multi_class="auto")
logistics_regression_pca.fit(principal_componenets, Y)

print("f1 score with PCA:", f1_score(Y, logistics_regression_pca.predict(principal_componenets), average="weighted"))
print("Accuracy with PCA:", logistics_regression_pca.score(principal_componenets, Y))
```

## Notes

- Replace `"Meanvariance"` with the correct feature name if your version of `feets` uses a different spelling (e.g., `"MeanVariance"` or `"Mean"`, etc).
- Ensure that all features in `feature_names` are supported by your version of `feets` (use `feets.FeatureSpace.features` to list available features).
- CSV files must be available in the working directory.
- This workflow can be adapted for other time-series feature extraction or classification tasks.

---