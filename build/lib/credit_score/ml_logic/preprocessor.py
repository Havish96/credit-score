import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocess features (X) for training and testing
    """
    oe_cols = ['Payment_of_Min_Amount','Credit_Mix']
    mms_cols = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_Credit_Inquiries', 'Num_of_Delayed_Payment']
    ss_cols = ['Changed_Credit_Limit', 'Credit_History_Age']
    rs_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Num_of_Loan',
               'Delay_from_due_date', 'Outstanding_Debt', 'Monthly_Balance']

    print("\n🛠️ Preprocessing features ...")

    X_processed = ordinal_encode(X, oe_cols)
    X_processed = min_max_scale(X_processed, mms_cols)
    X_processed = standard_scale(X_processed, ss_cols)
    X_processed = robust_scale(X_processed, rs_cols)

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed

def preprocess_target(y: pd.Series) -> np.ndarray:
    """
    Preprocess target (y) for training and testing
    """
    print("\n🛠️ Preprocessing target ...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("✅ y_encoded, with shape", y_encoded.shape)
    print("Classes:", le.classes_)

    return y_encoded

def ordinal_encode(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Ordinal encode columns
    """
    print(f"\nProcessing column: {cols} with ordinal encoding ...")
    X_encoded = X.copy()
    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_ordinal = pd.DataFrame(ordinal_encoder.fit_transform(X[cols]), index = X.index,
                          columns = cols)

    X_encoded[cols] = X_ordinal[cols]

    return X_encoded

def min_max_scale(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Min-max scale columns
    """
    print(f"\nProcessing column: {cols} with min-max-scaler ...")
    X_scaled = X.copy()
    min_max_scaler = MinMaxScaler()

    X_scaled[cols] = min_max_scaler.fit_transform(X[cols])

    return X_scaled

def standard_scale(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Standard scale columns
    """
    print(f"\nProcessing column: {cols} with standard-scaler ...")
    X_scaled = X.copy()
    standard_scaler = StandardScaler()

    X_scaled[cols] = standard_scaler.fit_transform(X[cols])

    return X_scaled

def robust_scale(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Robust scale columns
    """
    print(f"\nProcessing column: {cols} with robust-scaler ...")
    X_scaled = X.copy()
    robust_scaler = RobustScaler()

    X_scaled[cols] = robust_scaler.fit_transform(X[cols])

    return X_scaled
