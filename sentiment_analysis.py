# sentiment_analysis.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # or any other pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 classes: positive, negative, neutral

# Function to analyze sentiment using BERT
def analyze_sentiment(text):
    # Tokenize input text and add special tokens [CLS] and [SEP]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted logits
    logits = outputs.logits
    
    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=1)
    
    # Get predicted sentiment label (class with highest probability)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Map predicted class to sentiment label
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment = sentiment_labels[predicted_class]
    
    return sentiment
