import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.xlsx")
FEATURES = ["Moist %", "Ash %"]
TARGET = "Eq. M%"

_state = {"models": None, "metrics": None, "scaler": None}


def _load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = [" ".join(c.split()) for c in df.columns]  # collapse whitespace
    df = df[["Moist %", "Ash %", "Eq. M%"]].dropna()
    return df


def _metrics(y_true, y_pred):
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def train_all():
    """Train all 5 models on the full dataset and cache them in memory."""
    df = _load_data()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}
    metrics = {}

    # 1. XGBoost (HPT-tuned)
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        subsample=1.0,
        n_estimators=500,
        min_child_weight=10,
        max_depth=4,
        learning_rate=0.03,
        gamma=0.3,
        colsample_bytree=1.0,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model
    metrics["XGBoost"] = _metrics(y_test, xgb_model.predict(X_test))

    # 2. CatBoost
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=4,
        loss_function="RMSE",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    models["CatBoost"] = cat_model
    metrics["CatBoost"] = _metrics(y_test, cat_model.predict(X_test))

    # 3. LightGBM
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "num_leaves": 14,
        "learning_rate": 0.02698175343227787,
        "max_depth": 14,
        "min_child_samples": 13,
        "subsample": 0.7553061858137803,
        "colsample_bytree": 0.9440765631254832,
        "n_estimators": 228,
    }
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    models["LightGBM"] = lgb_model
    metrics["LightGBM"] = _metrics(y_test, lgb_model.predict(X_test))

    # 4. Darts Hybrid (Gradient Boosting Regressor)
    darts_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )
    darts_model.fit(X_train, y_train)
    models["Darts Hybrid"] = darts_model
    metrics["Darts Hybrid"] = _metrics(y_test, darts_model.predict(X_test))

    # 5. TensorFlow Probability (feed-forward NN, scaled inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tfp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    tfp_model.fit(X_train_scaled, y_train)
    models["TensorFlow Probability"] = tfp_model
    metrics["TensorFlow Probability"] = _metrics(
        y_test, tfp_model.predict(X_test_scaled)
    )

    _state["models"] = models
    _state["metrics"] = metrics
    _state["scaler"] = scaler
    return models, metrics


def get_state():
    if _state["models"] is None:
        train_all()
    return _state


def predict_all(moisture, ash, gcv):
    state = get_state()
    X_input = pd.DataFrame([[moisture, ash]], columns=FEATURES)
    results = {}
    for name, model in state["models"].items():
        if name == "TensorFlow Probability":
            X_scaled = state["scaler"].transform(X_input)
            pred = float(model.predict(X_scaled)[0])
        else:
            pred = float(model.predict(X_input)[0])

        calculated_gcv = gcv * (100 - pred) / (100 - moisture)

        results[name] = {
            "eq_moisture": round(pred, 3),
            "gcv": round(calculated_gcv, 3),
        }
    return results
