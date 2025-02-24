import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
y = np.log1p(train_data["SalePrice"])
X = train_data.drop(columns=["SalePrice", "Id"])
test_ids = test_data["Id"]
test_features = test_data.drop(columns=["Id"])

numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns


def feature_engineering(data):
    data = data.copy()
    data['Overall_GrLivArea'] = data['OverallQual'] * data['GrLivArea']
    data['Overall_LotArea'] = data['OverallQual'] * data['LotArea']
    return data

X = feature_engineering(X)
test_features = feature_engineering(test_features)

def handle_missing(data):
    data = data.copy()
    for col in numeric_features:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_features:
        data[col] = data[col].fillna("None")
    return data

X = handle_missing(X)
test_features = handle_missing(test_features)

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
])

# Categorical transformation pipeline with OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(pipeline, X_valid, y_valid):
    y_pred = pipeline.predict(X_valid)
    return np.sqrt(mean_squared_error(y_valid, y_pred))

models = {
    "Lasso": Lasso(alpha=0.001)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    rmse = evaluate_model(pipeline, X_valid, y_valid)
    results[name] = rmse
    print(f"{name}: RMSE = {rmse:.4f}")
#Lasso: RMSE = 0.1379

best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

# Final model training on all data
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
final_pipeline.fit(X, y)

# Test set prediction
test_predictions = np.expm1(final_pipeline.predict(test_features))

# Prepare the submission file
submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_predictions})
submission.to_csv("submission.csv", index=False)
