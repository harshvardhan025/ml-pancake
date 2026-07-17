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
# Used Car Price Prediction (Techniques covered)
# This example includes:
# 1. Small real-world style dataset
# 2. Data cleaning
# 3. Feature and target separation
# 4. Train-test split
# 5. Handling numerical and categorical columns
# 6. Missing value handling
# 7. Feature scaling
# 8. One-hot encoding
# 9. Pipeline
# 10. Linear Regression
# 11. Ridge and Lasso comparison
# 12. Hyperparameter tuning using GridSearchCV
# 13. Model evaluation using MAE, MSE, RMSE, and R²
# 14. Predicting new data
# 15. Saving and loading the trained model
# -----------------------------------------------------------------------
# train_test_split is used to split arrays/dataframes into random train and test subsets, 
# and random_state helps make results reproducible. StandardScaler standardizes numeric features 
# by removing the mean and scaling to unit variance.
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
# -----------------------------------------------------------------------
# 1. Create a Small Real-World Style Dataset
# -----------------------------------------------------------------------
data = {
    "car_age": [1, 2, 3, 4, 5, 6, 2, 7, 8, 3, 4, 5, 6, 1, 9],
    "km_driven": [10000, 20000, 30000, 40000, 50000, 60000, 15000, 70000, 80000, 25000, 45000, 55000, 65000, 12000, 90000],
    "fuel_type": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Petrol", "Diesel"],
    "transmission": ["Manual", "Manual", "Automatic", "Manual", "Manual", "Automatic", "Automatic", "Manual", "Manual", "Automatic", "Manual", "Automatic", "Manual", "Automatic", "Manual"],
    "owner_count": [1, 1, 1, 2, 2, 2, 1, 3, 3, 1, 2, 2, 3, 1, 3],
    "price": [800000, 720000, 700000, 610000, 550000, 500000, 780000, 430000, 390000, 690000, 570000, 510000, 460000, 820000, 350000]
}

df = pd.DataFrame(data)

print(df)
# -----------------------------------------------------------------------
# 2. Basic Data Check
# -----------------------------------------------------------------------
print("Dataset shape:", df.shape)
print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())
# -----------------------------------------------------------------------
# 3. Separate Features and Target
# Here:
# X = input features
# y = output/target value
# -----------------------------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# -----------------------------------------------------------------------
# 4. Train-Test Split
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# -----------------------------------------------------------------------
# 5. Define Numeric and Categorical Columns
# -----------------------------------------------------------------------
numeric_features = ["car_age", "km_driven", "owner_count"]
categorical_features = ["fuel_type", "transmission"]

# -----------------------------------------------------------------------
# 6. Create Preprocessing Pipelines
# -----------------------------------------------------------------------
# For numeric columns:

# ➡️ Fill missing values with median
# ➡️ Scale data using StandardScaler

# For categorical columns:

# ➡️ Fill missing values with most frequent value
# ➡️ Convert text to numbers using OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# -----------------------------------------------------------------------
# 7. Create Preprocessing Pipelines
# -----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------------------------------------------------
# 8. Build Model Pipeline
# -----------------------------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# -----------------------------------------------------------------------
# 9. Train Linear Regression Model
# -----------------------------------------------------------------------
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------
# 10. Make Predictions
# -----------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# -----------------------------------------------------------------------
# 11. Evaluate the Model
# -----------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Meaning of metrics:-
# 1. MAE  = Average absolute prediction error
# 2. MSE  = Average squared prediction error
# 3. RMSE = Error in same unit as price
# 4. R2   = How well the model explains price variation

# -----------------------------------------------------------------------
# 12. Evaluate the Model
# -----------------------------------------------------------------------
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(result)

# -----------------------------------------------------------------------
# 13. Use GridSearchCV for Model Selection
# -----------------------------------------------------------------------

# Now we compare:

# ➡️ Linear Regression
# ➡️ Ridge Regression
# ➡️ Lasso Regression

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

param_grid = [
    {
        "model": [LinearRegression()],
        "model__fit_intercept": [True, False]
    },
    {
        "model": [Ridge()],
        "model__alpha": [0.1, 1.0, 10.0]
    },
    {
        "model": [Lasso(max_iter=10000)],
        "model__alpha": [0.01, 0.1, 1.0]
    }
]

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error"
)

grid_search.fit(X_train, y_train)

print("Best Model:", grid_search.best_estimator_)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# -----------------------------------------------------------------------
# 14. Evaluate Best Model
# -----------------------------------------------------------------------
best_model = grid_search.best_estimator_

best_predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, best_predictions)
mse = mean_squared_error(y_test, best_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, best_predictions)

print("Best Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------------------------------------------------
# 15. Evaluate Best Model
# -----------------------------------------------------------------------
new_car = pd.DataFrame({
    "car_age": [3],
    "km_driven": [28000],
    "fuel_type": ["Petrol"],
    "transmission": ["Automatic"],
    "owner_count": [1]
})

predicted_price = best_model.predict(new_car)

print("Predicted Car Price:", predicted_price[0])

# -----------------------------------------------------------------------
# 16. Save the Trained Model
# -----------------------------------------------------------------------
joblib.dump(best_model, "used_car_price_model.pkl")

print("Model saved successfully.")

# -----------------------------------------------------------------------
# 17. Load the Model and Predict Again
# -----------------------------------------------------------------------
loaded_model = joblib.load("used_car_price_model.pkl")

new_prediction = loaded_model.predict(new_car)

print("Prediction from loaded model:", new_prediction[0])

# -----------------------------------------------------------------------
# Complete Flow Summary
# -----------------------------------------------------------------------
# 1. Create dataset
# 2. Check data
# 3. Separate X and y
# 4. Split into train/test
# 5. Preprocess numeric columns
# 6. Preprocess categorical columns
# 7. Build pipeline
# 8. Train Linear Regression model
# 9. Predict on test data
# 10. Evaluate model
# 11. Compare Linear, Ridge, and Lasso
# 12. Tune using GridSearchCV
# 13. Predict new car price
# 14. Save/load model

