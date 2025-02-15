import os
import re
import random
import json
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load JSON intents with error handling
json_path = 'C:/Users/sudar/Desktop/chatbot/dataset/mentalhealth.json'
try:
    with open(json_path, 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Intents file not found. Please check the file path.")

# Load dataset
csv_path = 'C:/Users/sudar/Desktop/chatbot/dataset/mentalhealth.csv'
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Please check the file path.")
except pd.errors.EmptyDataError:
    raise ValueError("The dataset file is empty.")

# Preprocess dataset
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

questions = data['Questions'].apply(preprocess_text).tolist()

# Use Pandas category dtype for memory efficiency
data['Answers'] = data['Answers'].astype('category')
encoded_answers = data['Answers'].cat.codes

# Load BERT model and tokenizer
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['Answers'].cat.categories))
bert_model.eval()

# Tokenize and pad sequences
def encode_texts(texts, max_len=32):
    encodings = tokenizer(texts, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

x_encodings, x_attention_masks = encode_texts(questions)

# Convert to NumPy arrays
x_encodings = x_encodings.numpy()
x_attention_masks = x_attention_masks.numpy()

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x_encodings, encoded_answers, test_size=0.2, random_state=42)
attention_train, attention_test = train_test_split(x_attention_masks, test_size=0.2, random_state=42)

# Train chatbot model
optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Rule-Based Response Handling
def get_rule_based_response(user_input):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if user_input.lower() == pattern.lower():
                return random.choice(intent['responses'])
    return None

# BERT-Based Response Retrieval
def bert_response(user_input):
    try:
        encodings = tokenizer(user_input, max_length=32, truncation=True, padding='max_length', return_tensors='pt')
        
        with torch.no_grad():
            output = bert_model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            predicted_label = torch.argmax(output.logits, dim=1).item()
        
        return data['Answers'].cat.categories[predicted_label] if predicted_label < len(data['Answers'].cat.categories) else "I'm sorry, I couldn't understand your question."
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Chat Function
print("Mental Health Chatbot Ready!")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye! Take care.")
        break
    response = get_rule_based_response(user_input) or bert_response(user_input)
    print("Bot:", response)
