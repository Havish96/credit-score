import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

from credit_score.ml_logic.preprocessor import preprocess_features
from credit_score.ml_logic.model import load_model

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(Age: float,
            Payment_of_Min_Amount: str,
            Credit_Mix: str,
            Num_Bank_Accounts: float,
            Num_Credit_Card: float,
            Num_Credit_Inquiries: float,
            Num_of_Delayed_Payment: float,
            Changed_Credit_Limit: float,
            Credit_History_Age: float,
            Annual_Income: float,
            Monthly_Inhand_Salary: float,
            Interest_Rate: float,
            Num_of_Loan: float,
            Delay_from_due_date: float,
            Outstanding_Debt: float,
            Monthly_Balance: float):


    my_dict = {'Age': Age,
               'Payment_of_Min_Amount': Payment_of_Min_Amount,
               'Credit_Mix': Credit_Mix,
               'Num_Bank_Accounts': Num_Bank_Accounts,
               'Num_Credit_Card': Num_Credit_Card,
               'Num_Credit_Inquiries': Num_Credit_Inquiries,
               'Num_of_Delayed_Payment': Num_of_Delayed_Payment,
               'Changed_Credit_Limit': Changed_Credit_Limit,
               'Credit_History_Age': Credit_History_Age,
               'Annual_Income': Annual_Income,
               'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
               'Interest_Rate': Interest_Rate,
               'Num_of_Loan': Num_of_Loan,
               'Delay_from_due_date': Delay_from_due_date,
               'Outstanding_Debt': Outstanding_Debt,
               'Monthly_Balance': Monthly_Balance
               }

    X = pd.DataFrame(my_dict, index=[0])
    X_processed = preprocess_features(X)

    pred = float(app.state.model.predict(X_processed))
    score = ''

    if pred == 0:
        score = 'Poor'
    elif pred == 1:
        score = 'Average'
    elif pred == 2:
        score = 'Good'

    return {'Credit_Score': score}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
