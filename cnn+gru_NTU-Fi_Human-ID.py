import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

import os
from scipy.io import loadmat

# === Config ===

ROOT = "./NTU-Fi-HumanID"
BATCH_SIZE = 64
NUM_CLASSES = 14
NUM_EPOCHS = 100
LR = 1e-3

CLASS_NAMES = [f"Subject {i}" for i in range(1, NUM_CLASSES + 1)]

# Folder IDs actually present: 001–013, 015 (no 014)
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
assert len(SUBJECT_IDS) == NUM_CLASSES

DEVICE = "cpu"

# === Load all .mat files for a split ("train" or "test") ===

def load_split(split_dir, subject_ids):
    X_list = []
    y_list = []

    for label_idx, sid in enumerate(subject_ids):
        # folder like 001, 002, ..., 015
        full_dir = os.path.join(split_dir, f"{sid:03d}")

        for fname in sorted(os.listdir(full_dir)):
            if not fname.endswith(".mat"):
                continue

            mf = os.path.join(full_dir, fname)
            mat = loadmat(mf)

            # first non-metadata key (like your reference function)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if not keys:
                raise ValueError(f"No data key found in {mf}")
            data = mat[keys[0]]  # expected (N, T, F) or (T, F)

            # handle single vs multiple samples per file
            if data.ndim == 2:
                # single sample -> (1, T, F)
                data = data[None, :, :]
            elif data.ndim != 3:
                raise ValueError(f"Unexpected shape {data.shape} in {mf}")

            X_list.append(data)
            # label_idx: 0..NUM_CLASSES-1
            y_list.append(np.full(data.shape[0], label_idx, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

# === Load NTU-Fi Human-ID data from folder ===

train_dir = os.path.join(ROOT, "test_amp") # datasets somehow are flipped
val_dir   = os.path.join(ROOT, "train_amp")

X_train, y_train = load_split(train_dir, SUBJECT_IDS)
X_val,   y_val   = load_split(val_dir,   SUBJECT_IDS)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()

X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.long).squeeze()

# Data was still reversed for some reason
X_train = X_train.permute(0, 2, 1)  # now (546, 2000, 342)
X_val   = X_val.permute(0, 2, 1)    # now (294, 2000, 342)

# --- Downsample in time: keep every 4th frame (2000 -> 500) ---
X_train = X_train[:, ::4, :]        # (B, 500, 342)
X_val   = X_val[:, ::4, :]          # (B, 500, 342)

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:  ", y_val.shape)

# Create datasets & loaders
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# === CNN+GRU ===
#    (expects input as (B, T, F) = (batch, time, features)

class CNNGRU(nn.Module):
    def __init__(
        self,
        in_channels,        # feature_dim = 90
        num_classes=NUM_CLASSES,
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

# === Training / Evaluation Utilities ===

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

# === Model & train ===

# figure out input feature size from X_train: (N, T, F)
_, T, F = X_train.shape
print("Sequence length:", T, "Feature dim:", F)

model = CNNGRU(in_channels=F, num_classes=NUM_CLASSES).to(DEVICE)
#print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs,  val_accs  = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc     = eval_epoch(model, val_dl, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )

# === Post-training evaluation on validation set ===

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        outputs = model(x)                  # (B, num_classes)
        preds = outputs.argmax(dim=1)       # (B,)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# --- Overall metrics ---
acc = accuracy_score(all_labels, all_preds)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0
)

print("\n=== Overall Validation Metrics ===")
print(f"Accuracy:           {acc:.4f}")
print(f"Macro Precision:    {precision_macro:.4f}")
print(f"Macro Recall:       {recall_macro:.4f}")
print(f"Macro F1-Score:     {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1-Score:  {f1_weighted:.4f}")

# --- Class-wise metrics ---
print("\n=== Class-wise Metrics (Validation) ===")
report = classification_report(
    all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
)
print(report)

# Raw arrays if you want them programmatically
prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0
)
for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name}: precision={prec_cls[i]:.4f}, "
        f"recall={rec_cls[i]:.4f}, f1={f1_cls[i]:.4f}, support={support_cls[i]}"
    )

# --- Confusion matrix & heatmap ---
cm = confusion_matrix(all_labels, all_preds)

# Normalize rows to sum to 1
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues_r", vmin=0, vmax=1)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
plt.yticks(tick_marks, CLASS_NAMES)

fmt = ".2f"
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, format(cm_norm[i, j], fmt),
                 ha="center", va="center", fontsize=6, # reduced to 6
                 color="white" if cm_norm[i, j] < thresh else "black")

plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# === Training curves ===

epochs = range(1, NUM_EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs,   label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
