FROM python:3.8.12-buster

WORKDIR /Users/havish/code/Havish96/credit-score

# First, pip install dependencies
COPY prod_requirements.txt prod_requirements.txt
RUN pip install -r prod_requirements.txt

# Then only, install taxifare!
COPY credit_score credit_score
COPY models models
COPY encoders encoders
COPY setup.py setup.py
RUN pip install .

CMD uvicorn credit_score.api.fast:app --host 0.0.0.0 --port $PORT
