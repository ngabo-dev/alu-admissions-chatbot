from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import random
import json
import torch
import asyncio

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = FastAPI(title="ALU University Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../web"), name="static")

# Serve static files directly
@app.get("/styles.css")
async def get_styles():
    return FileResponse('../web/styles.css', media_type='text/css')

@app.get("/script.js")
async def get_script():
    return FileResponse('../web/script.js', media_type='application/javascript')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the processed dataset (originally from Hugging Face)
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "../models/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ALU Assistant"

class ChatRequest(BaseModel):
    message: str

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand. Could you please rephrase your question?"

@app.get("/")
async def read_root():
    return FileResponse('../web/index.html')

@app.get("/styles.css")
async def get_styles():
    return FileResponse('../web/styles.css', media_type='text/css')

@app.get("/script.js")
async def get_script():
    return FileResponse('../web/script.js', media_type='application/javascript')

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Add a small delay to simulate thinking
    await asyncio.sleep(1.5)

    response = get_response(request.message)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)