# -----------------------------------------------------------------------------------------------------------
# What is a Decision Tree?
# A Decision Tree makes decisions by asking questions.
# Example:
# Hours Studied > 5?
#       |
#   Yes | No
#       |
#     Pass
#       |
# Attendance > 70?
#       |
#   Yes | No
#       |
#    Pass Fail
# -----------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------
# Dataset
# --------------------

# [Hours Studied, Attendance]
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

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# --------------------
# Split Dataset
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------
# Create Model
# --------------------

model = DecisionTreeClassifier(
    random_state=42
)

# --------------------
# Train Model
# --------------------

model.fit(X_train, y_train)

# --------------------
# Make Predictions
# --------------------

y_pred = model.predict(X_test)

# --------------------
# Accuracy
# --------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# --------------------
# New Prediction
# --------------------

new_student = [[6.5, 78]]

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("\nPrediction:", prediction[0])

print("Probability [Fail, Pass]:")
print(probability)

# --------------------
# Feature Importance
# --------------------

features = ["Hours", "Attendance"]

print("\nFeature Importance:")

for name, score in zip(features, model.feature_importances_):
    print(name, ":", round(score, 3))

# --------------------
# Confusion Matrix
# --------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# --------------------
# Classification Report
# --------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
