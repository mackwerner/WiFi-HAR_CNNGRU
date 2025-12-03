import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# =================================
# Load UT-HAR data from folder
# =================================

ROOT = "/Users/mack/Desktop/Computer Vision/CV Final Project/UT-HAR Dataset" # Change to your folder

X_train = np.load(f"{ROOT}/data/X_train.csv", allow_pickle=True)
y_train = np.load(f"{ROOT}/label/y_train.csv", allow_pickle=True)

X_val   = np.load(f"{ROOT}/data/X_val.csv", allow_pickle=True)
y_val   = np.load(f"{ROOT}/label/y_val.csv", allow_pickle=True)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()

X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.long).squeeze()

print("X_train:", X_train.shape)
print("X_val:",   X_val.shape)
print("y_train:", y_train.shape)
print("y_val:",   y_val.shape)

# Create datasets & loaders
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)

# =================================
# CNN+GRU model for UT-HAR
#    (expects input as (B, T, F) = (batch, time, features)
# =================================

class CNNGRU(nn.Module):
    def __init__(
        self,
        in_channels,        # this will be feature_dim = 90
        num_classes=7,      # There are 7 classes, not 6 (0-6)
        cnn_channels=64,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.5,
        bidirectional=True,
    ):
        super(CNNGRU, self).__init__()

        # 1D CNN over time (we'll transpose to (B, C, T) inside forward)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.bidirectional = bidirectional
        gru_input_size = cnn_channels * 2

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,          # (B, T, F)
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        fc_in = gru_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x):
        # x: (B, T, F) = (batch, time, features)
        # transpose to (B, C, T) for Conv1d, where C = features
        x = x.permute(0, 2, 1)   # (B, F, T)

        x = self.cnn(x)          # (B, C', T')
        x = x.permute(0, 2, 1)   # (B, T', C')

        # GRU over time
        out, _ = self.gru(x)     # (B, T', H * num_directions)

        # take last timestep
        last = out[:, -1, :]     # (B, H * num_directions)

        last = self.dropout(last)
        logits = self.fc(last)   # (B, num_classes)
        return logits


# =================================
# Training / Evaluation Utilities
# =================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)  # shape (B, T, F)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)          # (B, num_classes)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# =================================
# Model & train
# =================================

# figure out input feature size from X_train: (N, T, F)
_, T, F = X_train.shape
print("Sequence length:", T, "Feature dim:", F)

model = CNNGRU(in_channels=F, num_classes=7).to(DEVICE)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc     = eval_epoch(model, val_dl, criterion)

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )
