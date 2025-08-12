import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# ========== CONFIG ==========
FEATURE_PATH = r"D:\Sajin Python Works\Sajin Work\Symbolic reasoning\sajin CT MRI\sajin CT MRI\fused_features.pt"
CBM_MODEL_PATH = r"D:\Sajin Python Works\Sajin Work\Symbolic reasoning\sajin CT MRI\sajin CT MRI\cbm_model.pth"
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set seed for reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ========== Dataset ==========
class ConceptDataset(Dataset):
    def __init__(self, feature_file):
        data = torch.load(feature_file)
        self.features = data["features"]
        self.labels = data["labels"]

        self.shape_labels = [(1 if label == 1 else 0) for label in self.labels]
        self.effect_labels = [(1 if label == 1 else 0) for label in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = torch.tensor([self.shape_labels[idx], self.effect_labels[idx]], dtype=torch.float)
        return x, y

# ========== Model ==========
class CBMModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_concepts=2):
        super(CBMModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_concepts)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.norm1(self.fc1(x))))
        x = self.dropout(self.relu(self.norm2(self.fc2(x))))
        return self.fc3(x)

# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ========== Accuracy ==========
def binary_accuracy(output, target):
    preds = (torch.sigmoid(output) > 0.5).float()
    correct = (preds == target).float()
    return correct.mean().item()

# ========== Training with Graph ==========
def train_cbm_with_graph():
    dataset = ConceptDataset(FEATURE_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CBMModel().to(DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc += binary_accuracy(out, y)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_acc += binary_accuracy(out, y)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CBM_MODEL_PATH)
            print(f"✅ Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    print(f"🏁 Best model saved to: {CBM_MODEL_PATH}")

    # ========== Graphs ==========
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("cbm_training_graph.png")
    plt.show()

if __name__ == "__main__":
    train_cbm_with_graph()
