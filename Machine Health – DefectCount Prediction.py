import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = "C://Users//User//Downloads//MACHINE HEALTH & MAINTENANCE.xlsx"
df = pd.read_excel(file_path)

print("Dataset Shape:", df.shape)
print(df.head())

features = [
    "Temperature",
    "Vibration",
    "Pressure",
    "EnergyConsumption",
    "ProductionUnits",
    "Plant",
    "MachineID"
]
target = "DefectCount"

X = df[features]
y = df[target]

categorical_cols = ["Plant", "MachineID"]
numerical_cols = [
    "Temperature",
    "Vibration",
    "Pressure",
    "EnergyConsumption",
    "ProductionUnits"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numerical", "passthrough", numerical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

print("\n----- Linear Regression Results -----")
print(f"MAE: {linear_mae:.3f}")
print(f"RMSE: {linear_rmse:.3f}")

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\n----- Random Forest Results -----")
print(f"MAE: {rf_mae:.3f}")
print(f"RMSE: {rf_rmse:.3f}")

encoded_feature_names = (
    rf_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = rf_model.named_steps["regressor"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": encoded_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n----- Top 10 Important Features -----")
print(feature_importance_df.head(10))