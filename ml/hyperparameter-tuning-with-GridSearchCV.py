# --------------------------------------------------------------------------
# What is Hyperparameter Tuning?
# When you create a model, some settings must be chosen before training. 
# These settings are called hyperparameters.
# --------------------------------------------------------------------------
# For example:
RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
# Here:
# n_estimators = Number of trees
# max_depth = Maximum depth of each tree
# These are hyperparameters because the model does not learn them automatically.
# --------------------------------------------------------------------------
# The Problem
# How do we know whether:
# n_estimators = 10 is better than n_estimators = 100
# or n_estimators = 200 
# Instead of trying manually:
# Try 10 trees
# Try 50 trees
# Try 100 trees
# Try 200 trees
# we use GridSearchCV.
# --------------------------------------------------------------------------
# What is GridSearchCV?
# GridSearchCV automatically tries all possible hyperparameter combinations and finds the best one.
# Example Grid
{
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 5, 10]
}
# GridSearchCV will try:
n_estimators=10,  max_depth=2
n_estimators=10,  max_depth=5
n_estimators=10,  max_depth=10

n_estimators=50,  max_depth=2
n_estimators=50,  max_depth=5
n_estimators=50,  max_depth=10

n_estimators=100, max_depth=2
n_estimators=100, max_depth=5
n_estimators=100, max_depth=10
# Total:
3 × 3 = 9 models
# It then picks the best-performing one.
# --------------------------------------------------------------------------
# Complete Example
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------
# Dataset
# --------------------

X = [
    [1, 50],
    [2, 55],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85],
    [9, 90],
    [10, 95]
]

y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# --------------------
# Split
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------
# Model
# --------------------

model = RandomForestClassifier(random_state=42)

# --------------------
# Hyperparameter Grid
# --------------------

param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 5, 10]
}

# --------------------
# Grid Search
# --------------------

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

# --------------------
# Best Parameters
# --------------------

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Score:")
print(grid_search.best_score_)

# --------------------
# Best Model
# --------------------

best_model = grid_search.best_estimator_

# --------------------
# Test Prediction
# --------------------

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(accuracy)

# Visual Representation
# --------------------------------------------------------------------------
#                    GridSearchCV
#
#                  Hyperparameters
#                         |
#       ------------------------------------
#       |                |                |
#   Trees=10           Trees=50          Trees=100
#   Depth=2            Depth=5           Depth=10
#       |                |                |
#     Train            Train            Train
#       |                |                |
#    85% Acc         92% Acc          95% Acc
#       |                |                |
#        -------------------------------
#                       |
#               Select Best Model
#                       |
#             Trees=100, Depth=10
# --------------------------------------------------------------------------
# Important Attributes
# 1. Best Parameters
# grid_search.best_params_
# Output:-
# {
#   'max_depth': 5,
#   'n_estimators': 100
# }
# 2. Best Accuracy
# grid_search.best_score_
# Output:-
# 0.95
# 3. Best Model
# grid_search.best_estimator_
# Returns:- 
# RandomForestClassifier(
#    max_depth=5,
#    n_estimators=100
# )
# --------------------------------------------------------------------------
# Common Hyperparameters to Tune
# [Logistic Regression]
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}
# [Random Forest]\
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}
# [Decision Tree]
param_grid = {
    "max_depth": [2, 4, 6, 8],
    "criterion": ["gini", "entropy"]
}
# [Support Vector Machine]
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}
# --------------------------------------------------------------------------
# ML Pipeline in Real Projects
# Load Data
#     ↓
# Clean Data
#     ↓
# Feature Engineering
#     ↓
# Train/Test Split
#     ↓
# Cross Validation
#     ↓
# GridSearchCV
#     ↓
# Best Model
#     ↓
# Evaluation
#     ↓
# Deployment
