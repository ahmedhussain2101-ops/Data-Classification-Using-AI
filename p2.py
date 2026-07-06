"""
Project 2: Data Classification Using AI
-----------------------------------------
Goal: Build a basic classification model using a small dataset.

Key Skills demonstrated:
- Data handling (loading, exploring a dataset)
- Supervised learning basics (features -> labels)
- Model training (train/test split, fitting, evaluation)

Dataset: Iris flower dataset (built into scikit-learn)
- 150 samples, 3 flower species (setosa, versicolor, virginica)
- 4 features: sepal length, sepal width, petal length, petal width
- A classic, beginner-friendly dataset for classification.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------
# STEP 1: Load and understand the dataset
# ---------------------------------------------------------------
iris = load_iris()

# Convert to a pandas DataFrame so it's easy to inspect and understand
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target  # 0, 1, 2 -> setosa, versicolor, virginica

print("=" * 60)
print("STEP 1: Understanding the dataset")
print("=" * 60)
print(f"Shape of dataset: {df.shape}  (rows, columns)")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution (how many samples per species):")
print(df["species"].value_counts())
print("\nBasic statistics:")
print(df.describe())

# ---------------------------------------------------------------
# STEP 2: Split data into features (X) and labels (y)
# X = the measurements the model learns from
# y = the correct answer (species) the model tries to predict
# ---------------------------------------------------------------
X = df[iris.feature_names]
y = df["species"]

# ---------------------------------------------------------------
# STEP 3: Split into training and testing sets
# Training set: used to teach the model
# Testing set: held back, used only to check performance afterwards
# test_size=0.2 -> 80% training, 20% testing
# random_state=42 -> makes the split reproducible every run
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("STEP 2 & 3: Train/test split")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ---------------------------------------------------------------
# STEP 4: Feature scaling
# KNN relies on distances between points, so features need to be
# on a similar scale (e.g. petal length in cm vs width in cm).
# We fit the scaler ONLY on training data to avoid data leakage.
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# STEP 5: Apply a simple classification algorithm
# K-Nearest Neighbors: classifies a new point based on the
# majority class among its 'k' closest neighbors in the training data.
# Simple, intuitive, and a great first classification algorithm.
# ---------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------
# STEP 6: Make predictions and evaluate performance
# ---------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\n" + "=" * 60)
print("STEP 4, 5 & 6: Model training and evaluation")
print("=" * 60)
print(f"Model used: K-Nearest Neighbors (k=5)")
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.2%}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# STEP 7: Predict a brand-new, unseen flower sample
# ---------------------------------------------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\n" + "=" * 60)
print("STEP 7: Predicting a new sample")
print("=" * 60)
print(f"Sample measurements: {sample[0]}")
print(f"Predicted species: {iris.target_names[prediction[0]]}")