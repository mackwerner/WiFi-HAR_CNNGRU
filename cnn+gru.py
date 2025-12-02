import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN + GRU model for UT-HAR
class CNNGRU(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=6,
        cnn_channels=64,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.5,
        bidirectional=True,
    ):
        super(CNNGRU, self).__init__()

        # 1D CNN over time
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
        # x: (B, C, T)
        x = self.cnn(x)          # (B, C', T')
        x = x.permute(0, 2, 1)   # (B, T', C')

        # GRU over time
        out, _ = self.gru(x)     # (B, T', H * num_directions)

        # take last timestep
        last = out[:, -1, :]     # (B, H * num_directions)

        last = self.dropout(last)
        logits = self.fc(last)   # (B, num_classes)
        return logits


# ============================
# Training / eval functions
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
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


# ============================
# Main training script
# ============================

# make_uthar_loaders should already be defined somewhere else
root = "/path/uthar_dataset"
train_loader, val_loader, test_loader = make_uthar_loaders(root, batch_size=64)

# figure out input channels from one batch
x_batch, _ = next(iter(train_loader))
_, C_in, _ = x_batch.shape
print("Example batch shape:", x_batch.shape)

model = CNNGRU(in_channels=C_in, num_classes=6).to(DEVICE)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )
