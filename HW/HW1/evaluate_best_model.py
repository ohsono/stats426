import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# 1. Generate Data (Same as notebook)
def generate_data(N=10000, p=100, noise_level=0.15, correlation=0.5):
    np.random.seed(42)
    torch.manual_seed(42)
    
    cov_matrix = np.full((p, p), correlation)
    np.fill_diagonal(cov_matrix, 1.0)
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, cov_matrix, size=N)
    beta = np.random.normal(0, 1, size=p)
    logits = X @ beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    n_flip = int(noise_level * N)
    flip_indices = np.random.choice(np.arange(N), size=n_flip, replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor

X, y = generate_data()

# 2. Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=6000, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

# 3. Model Definition
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        # Init
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.layer2(x)

# 4. Train Best Model
# Best params from grid search: Hidden=128, LR=0.01, Batch=32
H_DIM = 128
LR = 0.01
BATCH = 32
EPOCHS = 30

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

model = DynamicMLP(100, H_DIM)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

history = {'train_loss': [], 'val_loss': []}

print("Starting training of best model...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_ds)
    history['train_loss'].append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
    val_loss = val_loss / len(val_ds)
    history['val_loss'].append(val_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 5. Plot Learning Curves
plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('learning_curve.png')
print("Saved learning_curve.png")

# 6. Evaluate on Test Set
model.eval()
y_true = []
y_scores = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        y_true.extend(labels.numpy())
        y_scores.extend(probs.numpy())

test_auc = roc_auc_score(y_true, y_scores)
print(f"Test AUC: {test_auc:.4f}")

# 7. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("Saved roc_curve.png")
