from fastapi import FastAPI
from random import random

app = FastAPI()

@app.get("/")
def home():
    return "audio analysis page"

@app.get("/make-decision")
def analyze_audio():
    return {"probability_of_ai": random()}
