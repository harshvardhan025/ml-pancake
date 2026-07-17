# -----------------------------------------------------------------------
# A Random Forest is a collection of Decision Trees.
#
# Decision Tree 1
# Decision Tree 2
# Decision Tree 3
# Decision Tree 4
#       ...
#       ↓
# Majority Voting
#       ↓
# Final Prediction
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------
# 1. Dataset
# --------------------

# [Hours Studied, Attendance %]
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
# 2. Train/Test Split
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------
# 3. Create Model
# --------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# --------------------
# 4. Train Model
# --------------------

model.fit(X_train, y_train)

# --------------------
# 5. Make Predictions
# --------------------

y_pred = model.predict(X_test)

# --------------------
# 6. Accuracy
# --------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# --------------------
# 7. Predict New Student
# --------------------

new_student = [[6.5, 78]]

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("\nPrediction:", prediction[0])

print("Probability [Fail, Pass]:",
      probability[0])

# --------------------
# 8. Feature Importance
# --------------------

print("\nFeature Importance:")

features = ["Hours Studied", "Attendance"]

for feature, score in zip(features,
                          model.feature_importances_):
    print(feature, ":", round(score, 3))

# --------------------
# 9. Confusion Matrix
# --------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# --------------------
# 10. Classification Report
# --------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
