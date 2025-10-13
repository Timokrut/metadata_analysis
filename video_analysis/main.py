from fastapi import FastAPI
from random import random

app = FastAPI()

@app.get("/")
def home():
    return "video analysis page"

@app.get("/make-decision")
def analyze_video():
    return {"probability_of_ai": random()}

