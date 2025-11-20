from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import json
from sentence_transformers import SentenceTransformer, util
import os
import re

class Item(BaseModel):
    transaction : str

app = FastAPI()

@app.post("/predict")
async def predict(body : Item):
    body = preprocessing(body)
    return body
    
@app.post("/feedback")
async def feedback():
    return {"message":"Feedback"}


def preprocessing(body):
        # i am getting a transaction string in body, i want to remove all the special characters and convert it to lowercase
        body = body.transaction.lower()
        body = re.sub(r'[^a-z0-9 ]', '', body)
        return body