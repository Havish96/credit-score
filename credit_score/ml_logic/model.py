import xgboost as xgb
import numpy as np
import os
import time

def initialize_model() -> xgb.XGBClassifier:
    """
    Initialize the XGBClassifier model.
    """
    xgb_model = xgb.XGBClassifier(random_state = 42,
                                  learning_rate = 0.5,
                                  max_depth = 6,
                                  n_estimators = 200)

    return xgb_model

def train_model(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    xgb_model = model.fit(X, y)

    return xgb_model

def predict(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Make predictions on X using the fitted model.
    """
    y_pred = model.predict(X)

    return y_pred


def save_model(model: xgb.XGBClassifier) -> None:
    # """
    # Save the model locally.
    # """
    # timestamp = time.strftime("%Y%m%d-%H%M%S")

    # # Save model locally
    # model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    # model.save(model_path)

    # print("✅ Model saved locally")
    pass
