# -----------------------------------------------------------------------
# Starter Code
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

# Training data
# Hours studied
X = [[1], [2], [3], [4], [5], [6]]

# Result: 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Predict for a student who studied 3.5 hours
prediction = model.predict([[3.5]])

# Probability of fail/pass
probability = model.predict_proba([[3.5]])

print("Prediction:", prediction[0])
print("Probability [Fail, Pass]:", probability[0])

# -------------------------------------------------------
# Sample Output
# Prediction: 1
# Probability [Fail, Pass]: [0.42 0.58]\
#--------------------------------------------------------
# Meaning:
# 42% chance of Fail
# 58% chance of Pass
# Since Pass probability > 50%, prediction is 1 (Pass)
# -------------------------------------------------------

# -----------------------------------------------------------------------
# Complete Machine Learning workflow
# -----------------------------------------------------------------------
# it includes:- 
# ✅ Create dataset
# ✅ Split train/test data
# ✅ Train model
# ✅ Make predictions
# ✅ Check accuracy
# ✅ View probabilities
# ✅ Confusion Matrix
# ✅ Classification Report
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------
# 1. Dataset
# --------------------
X = [
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# --------------------
# 2. Split Dataset
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------
# 3. Train Model
# --------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# --------------------
# 4. Make Predictions
# --------------------
y_pred = model.predict(X_test)

# --------------------
# 5. Calculate Accuracy
# --------------------
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# --------------------
# 6. Predict New Data
# --------------------
new_student = [[5.5]]

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("\nPrediction:", prediction[0])
print("Probability [Fail, Pass]:", probability[0])

# --------------------
# 7. Confusion Matrix
# --------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# --------------------
# 8. Classification Report
# --------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
