from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import json
from sentence_transformers import SentenceTransformer, util
from .engine import FinScribeEngine
import os
import re

app = FastAPI()
engine = FinScribeEngine()

class TransactionRequest(BaseModel):
    transaction : str

@app.post("/predict")
async def predict(request : TransactionRequest):
    result = engine.predict(request.transaction)
    return result
    
@app.post("/feedback")
async def feedback():
    return {"message":"Feedback"}


def preprocessing(body):
        # i am getting a transaction string in body, i want to remove all the special characters and convert it to lowercase
        body = body.transaction.lower()
        body = re.sub(r'[^a-z0-9 ]', '', body)
        return body