import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "car_price_prediction_.csv"   # change path if needed
df = pd.read_csv(file_path)

# Define features and target
X = df.drop(columns=["Price", "Car ID"])  # Drop target + ID column
y = df["Price"]

# Select numerical and categorical features
num_features = ["Year", "Engine Size", "Mileage"]
cat_features = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

# Transformers
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# Build pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
