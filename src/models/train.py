import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Dataset Class
class EmotionEmbeddingDataset(Dataset):
    def __init__(self, metadata_df, embedding_dir, label_map):
        # Reset index so iloc matches current dataframe length
        self.df = metadata_df.reset_index(drop=True)
        self.embedding_dir = Path(embedding_dir)
        self.label_map = label_map
        
        # Pre-filter to only files that exist in the embedding dir
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            stem = Path(row['path']).stem
            # This matches the naming in build_features.py
            emb_path = self.embedding_dir / f"{row['split']}_{row['emotion']}_{stem}.npy"
            if emb_path.exists():
                self.valid_indices.append((idx, str(emb_path)))
        
        print(f"Loaded {len(self.valid_indices)} valid embeddings.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx, emb_path = self.valid_indices[idx]
        embedding = np.load(emb_path)
        label_str = self.df.iloc[original_idx]['emotion']
        label = self.label_map[label_str]
        
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 2. Simple Neural Network Architecture
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_model():
    # Setup paths
    METADATA = "data/processed/metadata.csv"
    EMB_DIR = "data/embeddings/wav2vec2"
    MODEL_SAVE_PATH = "models/emotion_classifier.pth"
    
    # Label Mapping
    df = pd.read_csv(METADATA)
    emotions = sorted(df['emotion'].unique())
    label_map = {name: i for i, name in enumerate(emotions)}
    print(f"Label Map: {label_map}")

    # Prepare DataLoaders
    train_ds = EmotionEmbeddingDataset(df[df['split']=='train'], EMB_DIR, label_map)
    val_ds = EmotionEmbeddingDataset(df[df['split']=='val'], EMB_DIR, label_map)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(num_classes=len(emotions)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 50
    best_val_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_map': label_map,
                'emotions': emotions
            }, MODEL_SAVE_PATH)
            print(f"Saved new best model!")

    print(f"\nTraining Complete. Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
