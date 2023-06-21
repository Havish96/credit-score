import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
import pickle

from credit_score.ml_logic.data import clean_data

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocess features (X) for training and testing
    """
    oe_cols = ['Payment_of_Min_Amount','Credit_Mix']
    mms_cols = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_Credit_Inquiries', 'Num_of_Delayed_Payment']
    ss_cols = ['Changed_Credit_Limit', 'Credit_History_Age']
    rs_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Num_of_Loan',
               'Delay_from_due_date', 'Outstanding_Debt', 'Monthly_Balance']

    encoders = load_encoders()
    X_processed = X.copy()

    print("\n🛠️ Preprocessing features ...")

    X_processed[oe_cols] = encoders[0].transform(X[oe_cols])
    X_processed[mms_cols] = encoders[1].transform(X_processed[mms_cols])
    X_processed[ss_cols] = encoders[2].transform(X_processed[ss_cols])
    X_processed[rs_cols] = encoders[3].transform(X_processed[rs_cols])

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed

def preprocess_target(y: pd.Series) -> np.ndarray:
    """
    Preprocess target (y) for training and testing
    """
    print("\n🛠️ Preprocessing target ...")

    y_encoded = y.map({'Poor': 0, 'Standard': 1, 'Good': 2})

    print("✅ y_encoded, with shape", y_encoded.shape)
    print("0 - Poor, 1 - Standard, 2 - Good")

    return y_encoded

def load_encoders() -> list:
    path = '/Users/havish/code/Havish96/credit-score/encoders/'

    with open(path + 'ordinal_encoder.pkl', 'rb') as file:
        ordinal_encoder = pickle.load(file)

    with open(path + 'min_max_scaler.pkl', 'rb') as file:
        min_max_scaler = pickle.load(file)

    with open(path + 'standard_scaler.pkl', 'rb') as file:
        standard_scaler = pickle.load(file)

    with open(path + 'robust_scaler.pkl', 'rb') as file:
        robust_scaler = pickle.load(file)

    return [ordinal_encoder, min_max_scaler, standard_scaler, robust_scaler]

def save_ordinal_encoder(X: pd.DataFrame, cols: list) -> OrdinalEncoder():
    """
    Ordinal encode columns
    """
    file_path = '/Users/havish/code/Havish96/credit-score/encoders/ordinal_encoder.pkl'

    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ordinal_encoder.fit(X[cols])

    with open(file_path, 'wb') as file:
        pickle.dump(ordinal_encoder, file)

def save_min_max_scaler(X: pd.DataFrame, cols: list) -> MinMaxScaler():
    """
    Min-max scale columns
    """
    file_path = '/Users/havish/code/Havish96/credit-score/encoders/min_max_scaler.pkl'

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X[cols])

    with open(file_path, 'wb') as file:
        pickle.dump(min_max_scaler, file)

def save_standard_scaler(X: pd.DataFrame, cols: list) -> StandardScaler():
    """
    Standard scale columns
    """
    file_path = '/Users/havish/code/Havish96/credit-score/encoders/standard_scaler.pkl'

    standard_scaler = StandardScaler()
    standard_scaler.fit(X[cols])

    with open(file_path, 'wb') as file:
        pickle.dump(standard_scaler, file)

def save_robust_scaler(X: pd.DataFrame, cols: list) -> RobustScaler():
    """
    Robust scale columns
    """
    file_path = '/Users/havish/code/Havish96/credit-score/encoders/robust_scaler.pkl'

    robust_scaler = RobustScaler()
    robust_scaler.fit(X[cols])

    with open(file_path, 'wb') as file:
        pickle.dump(robust_scaler, file)
