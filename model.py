# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, BucketIterator

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

# Example usage:
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

# Train the model (example)
for epoch in range(num_epochs):
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label.float())
        loss.backward()
        optimizer.step()

# Predict sentiment for a new text
text = "This movie is great!"
tokenized_text = [TEXT.vocab.stoi[token] for token in text.split()]
tensor_text = torch.LongTensor(tokenized_text).unsqueeze(1)
prediction = torch.sigmoid(model(tensor_text))
sentiment = 'positive' if prediction.item() >= 0.5 else 'negative'
print("Sentiment:", sentiment)
