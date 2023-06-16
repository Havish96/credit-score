import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Preprocess features (X) for training and testing
    """
    print("\n🛠️ Preprocessing features ...")

    X_processed = __process_type_of_loan(X)
    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X_processed)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed

def preprocess_target(y: pd.Series) -> np.ndarray:
    """
    Preprocess target (y) for training and testing
    """
    print("\n🛠️ Preprocessing target ...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("✅ y_encoded, with shape", y_encoded.shape)
    print("classes:", le.classes_)

    return y_encoded

def __process_type_of_loan(X: pd.DataFrame) -> pd.DataFrame:
    """
    Process 'Type_of_Loan' column into one-hot encoded columns
    """
    data_sep = ','
    col_sep = '_'

    object_cols = X.select_dtypes(include="object").columns
    dummy_cols   = [col for col in object_cols if X[col].str.contains(data_sep, regex=True).any()]
    dummy_prefix = [''.join(map(lambda x: x[0], col.split(col_sep))) if col_sep in col else col[:2] for col in dummy_cols]

    for col, pre in zip(dummy_cols, dummy_prefix):
        dummy_X = X.join(X[col].str.get_dummies(sep = data_sep).add_prefix(pre + col_sep))

    dummy_X.drop(columns = dummy_cols, inplace=True)
    columns = dummy_X.columns

    for col, pre in zip(dummy_cols, dummy_prefix):
        X_transformed = X.join(X[col].str.get_dummies(sep = data_sep).add_prefix(pre + col_sep))

    X_transformed = X_transformed.reindex(columns = columns, fill_value = 0)

    return X_transformed

def create_sklearn_preprocessor() -> ColumnTransformer:
    """
    Create a sklearn preprocessor pipeline for X features
    """
    one_hot_encode_cols = ['Occupation', 'Payment_Behaviour']
    ordinal_encode_cols = ['Payment_of_Min_Amount','Credit_Mix']
    min_max_scale_cols = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_Credit_Inquiries', 'Num_of_Delayed_Payment']
    standard_scale_cols = ['Credit_Utilization_Ratio', 'Changed_Credit_Limit', 'Credit_History_Age']
    robust_scale_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Num_of_Loan',
                         'Delay_from_due_date', 'Outstanding_Debt', 'Total_EMI_per_month',
                         'Amount_invested_monthly', 'Monthly_Balance']

    column_transformer = [('one_hot_encode', OneHotEncoder(sparse=False, handle_unknown='ignore'), one_hot_encode_cols),
                              ('ordinal_encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_encode_cols)]

    scaling_transformer = [('min_max_scale', MinMaxScaler(), min_max_scale_cols),
                           ('standard_scale', StandardScaler(), standard_scale_cols),
                           ('robust_scale', RobustScaler(), robust_scale_cols)]

    preprocessor = ColumnTransformer(transformers = column_transformer + scaling_transformer)

    pipeline = make_pipeline(preprocessor)

    return pipeline
