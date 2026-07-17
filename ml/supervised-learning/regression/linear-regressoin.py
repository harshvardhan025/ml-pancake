# -----------------------------------------------------------------------
# Starter Code
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
import numpy as np

# Input data (Hours Studied)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data (Marks)
y = np.array([35, 45, 50, 60, 70])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for a student who studied 6 hours
prediction = model.predict([[6]])

print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted Marks for 6 hours:", prediction[0])

# -----------------------------------------------------------------------
# Complete Machine Learning workflow
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# --------------------
# 1. Dataset
# --------------------

# Hours Studied
X = [
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
]

# Marks Obtained
y = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

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
model = LinearRegression()

model.fit(X_train, y_train)

# --------------------
# 4. Make Predictions
# --------------------
y_pred = model.predict(X_test)

# --------------------
# 5. Evaluate Model
# --------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# --------------------
# 6. Predict New Data
# --------------------
new_student = [[5.5]]

prediction = model.predict(new_student)

print("\nPredicted Marks:", prediction[0])

# --------------------
# 7. Model Details
# --------------------
print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# --------------------
# 8. Actual vs Predicted
# --------------------
print("\nActual Marks:", y_test)
print("Predicted Marks:", y_pred)

# -----------------------------------------------------------------------
# Real World Example
# -----------------------------------------------------------------------
