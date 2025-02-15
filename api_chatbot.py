from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Paths
MODEL_DIR = "C:/Users/sudar/Desktop/chatbot/models/"
TOKENIZER_NAME = "bert-base-uncased"
DATASET_PATH = "C:/Users/sudar/Desktop/chatbot/dataset/mentalhealth.csv"

# Load tokenizer and model
print("Loading tokenizer and trained model...")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
bert_model.eval()  # Set model to evaluation mode

# Load dataset
try:
    data = pd.read_csv(DATASET_PATH)
    data['Answers'] = data['Answers'].astype('category')  # Use category dtype for memory efficiency
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Please check the path.")

# Function to preprocess and encode user input
def encode_texts(texts, max_len=32):
    encodings = tokenizer(texts, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

@app.post("/chat/")
def chatbot_endpoint(user_input: str):
    try:
        # Encode input
        encoded_input, attention_mask = encode_texts([user_input])
        
        # Make prediction
        with torch.no_grad():
            output = bert_model(encoded_input, attention_mask=attention_mask)
            predicted_label = torch.argmax(output.logits, dim=1).item()

        # Get response
        response = data['Answers'].cat.categories[predicted_label] if predicted_label < len(data['Answers'].cat.categories) else "I'm sorry, I couldn't understand your question."
    
    except Exception as e:
        response = f"Error: {str(e)}"

    return {"response": response}

# API root endpoint
@app.get("/")
def root():
    return {"message": "Mental Health Chatbot API is running!"}
