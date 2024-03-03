# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, BucketIterator
from model import SentimentAnalysisModel

def train_model(train_data, valid_data, model, criterion, optimizer, num_epochs=5):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        # Training loop
        model.train()
        for batch in train_iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in valid_iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label.float())
                valid_loss += loss.item()

        # Calculate average loss
        train_loss /= len(train_iterator)
        valid_loss /= len(valid_iterator)

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'sentiment_model.pt')

# Example usage:
# Load and preprocess the dataset
# (Assuming train_data and valid_data are already prepared TabularDatasets)

# Initialize model
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
dropout = 0.5
model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)

# Load pre-trained word embeddings (e.g., GloVe)
TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100))
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
train_model(train_data, valid_data, model, criterion, optimizer, num_epochs=5)
