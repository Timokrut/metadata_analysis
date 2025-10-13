from fastapi import FastAPI 
import requests 
import os 
import json

app = FastAPI()

@app.get("/get-info")
def get_info():
    metadata_prob, audio_prob, video_prob = collect_probabilities()

    return {
        "final decision": "NOT AI" if ((metadata_prob + audio_prob + video_prob) / 3) > 0.3 else "AI",
        "avg_probability": (metadata_prob + audio_prob + video_prob) / 3 
    }

def collect_probabilities():
    metadata_prob = requests.get(os.getenv("META_LINK"))
    audio_prob = requests.get(os.getenv("AUDIO_LINK"))
    video_prob = requests.get(os.getenv("VIDEO_LINK"))
    
    md = json.loads(metadata_prob.text)["probability_of_ai"]
    au = json.loads(audio_prob.text)["probability_of_ai"]
    vi = json.loads(video_prob.text)["probability_of_ai"]

    return md, au, vi 

