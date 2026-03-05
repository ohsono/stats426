"""
Corrected code for MLP-1layer-classification.ipynb
Copy these cells into your notebook in order, running each one.
This will update all class definitions and fix the AttributeError.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# ============================================================================
# CELL 1: Define the SingleMLP class with _load_filter_data method
# ============================================================================

class SingleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SingleMLP, self).__init__()
        # Input Layer -> Hidden Layer
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # Hidden Layer -> Output (Scalar)
        self.output = nn.Linear(hidden_dim, 1)  # Binary output
        # Activation function
        self.activation = nn.ReLU()
        
        # Proper Initialization
        # Kaiming/He Init is standard for ReLU layers
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        
        # Xavier/Glorot Init is standard for the output layer
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _load_filter_data(path, name="Dataset"):
        """
        Loads the dataset from a CSV file, filters for digits 3 and 5, and prepares tensors.

        Args:
            path (str): The file path to the CSV dataset.
            name (str, optional): A name for the dataset (e.g., "Train", "Test") for logging purposes. 
                                  Defaults to "Dataset".

        Returns:
            tuple: A tuple containing two torch.Tensors:
                - X (torch.Tensor): Feature matrix of shape (N, 784) with float32 type.
                - y (torch.Tensor): Label vector of shape (N, 1) with float32 type, where 3->0 and 5->1.

        Raises:
            ValueError: If the filtered dataset is empty (i.e., no digits 3 or 5 found).
        """
        print(f"Loading {name} from {path}...")
        df = pd.read_csv(path)
    
        label_col = df.columns[0]
        
        # Digits 3 and 5
        target_digits = [3, 5]
        
        df_filtered = df[df[label_col].isin(target_digits)].copy()

        if df_filtered.empty:
            raise ValueError(f"Error: No data found for digits 3 and 5 in {path}. "
                           f"Please check if the file contains these labels in the first column ('{label_col}').")
            
        # Map 3 -> 0, 5 -> 1
        y = df_filtered[label_col].apply(lambda x: 0 if x == 3 else 1).values
        
        # Features. Drop label column.
        X = df_filtered.drop(columns=[label_col]).values
        
        # Ensure float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        print(f"  Shape after filtering: {X.shape}")
        
        return torch.tensor(X), torch.tensor(y).unsqueeze(1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        # We return logits (raw scores) because we use BCEWithLogitsLoss
        return self.output(x)

print("✓ SingleMLP class defined successfully with _load_filter_data method")

# ============================================================================
# CELL 2: Load and prepare the data
# ============================================================================

# Paths to your data files
train_path = './mnist_train.csv'
val_path = './mnist_val.csv'
test_path = './mnist_test.csv'

# Load Data using the static method in SingleMLP
X_train, y_train = SingleMLP._load_filter_data(train_path, "Train")
X_val, y_val = SingleMLP._load_filter_data(val_path, "Val")
X_test, y_test = SingleMLP._load_filter_data(test_path, "Test")

# Create TensorDatasets
BATCH_SIZE = 64
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

print(f"✓ Data loaded successfully")
print(f"  Train set: {len(train_ds)} samples")
print(f"  Val set: {len(val_ds)} samples")
print(f"  Test set: {len(test_ds)} samples")

# ============================================================================
# CELL 3: Define the run_experiment function (CORRECTED VERSION)
# ============================================================================

def run_experiment(config, train_ds, val_ds, test_ds):
    """
    Trains a model with specific hyperparameters and returns final metrics.
    Can accept either TensorDataset objects or file paths (strings).
    """
    # Unpack hyperparameter configuration dictionary
    i_dim = config['input_dim']
    h_dim = config['hidden_dim']
    lr = config['lr']
    bs = config['batch_size']
    epochs = config['epochs']

    # Helper to ensure input is a Dataset
    def ensure_dataset(data_input, name_suffix=""):
        if isinstance(data_input, str):
            # It's a path, load it
            X, y = SingleMLP._load_filter_data(data_input, name_suffix)
            return TensorDataset(X, y)
        return data_input

    # Prepare Datasets (Load if they are paths)
    train_ds = ensure_dataset(train_ds, "Train")
    val_ds = ensure_dataset(val_ds, "Val")
    test_ds = ensure_dataset(test_ds, "Test")

    # Create Local DataLoaders specific to this batch size (bs)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # Initialize Model & Optimizer
    model = SingleMLP(input_dim=i_dim, hidden_dim=h_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []
    
    print("\nStarting Training...")
    # --- Training Phase ---
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()           # Clear gradients
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights
            running_train_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = running_train_loss / len(train_ds)
        train_loss_history.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = running_val_loss / len(val_ds) 
        val_loss_history.append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mnist_experiment_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved loss plot to mnist_experiment_loss.png")

    # --- Validation Metrics (for model selection) ---
    model.eval()
    val_targets = []
    val_preds = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            val_targets.extend(labels.numpy())
            val_preds.extend(probs.numpy())

    avg_final_val_loss = val_loss_history[-1]
    
    # Calculate Validation Metrics
    val_auc = roc_auc_score(val_targets, val_preds)
    val_acc = accuracy_score(val_targets, (np.array(val_preds) > 0.5).astype(int))
    
    print(f"✓ Final Val Accuracy: {val_acc:.4f}")
    print(f"✓ Final Val AUC: {val_auc:.4f}")

    return avg_final_val_loss, val_auc

print("✓ run_experiment function defined successfully")

# ============================================================================
# CELL 4: Define experiment configurations - ASSIGNMENT REQUIREMENTS
# ============================================================================

# Grid Search Configuration
# Mini-batch size s ∈ {64, 128, 256}
# Hidden layer dimension h ∈ {64, 128, 256}
# This creates 3 × 3 = 9 total configurations

experiments = [
    # Hidden=64, Batch=64
    {'input_dim':784, 'hidden_dim': 64,  'lr': 0.01, 'batch_size': 64,  'epochs': 20},
    # Hidden=64, Batch=128
    {'input_dim':784, 'hidden_dim': 64,  'lr': 0.01, 'batch_size': 128, 'epochs': 20},
    # Hidden=64, Batch=256
    {'input_dim':784, 'hidden_dim': 64,  'lr': 0.01, 'batch_size': 256, 'epochs': 20},
    
    # Hidden=128, Batch=64
    {'input_dim':784, 'hidden_dim': 128, 'lr': 0.01, 'batch_size': 64,  'epochs': 20},
    # Hidden=128, Batch=128
    {'input_dim':784, 'hidden_dim': 128, 'lr': 0.01, 'batch_size': 128, 'epochs': 20},
    # Hidden=128, Batch=256
    {'input_dim':784, 'hidden_dim': 128, 'lr': 0.01, 'batch_size': 256, 'epochs': 20},
    
    # Hidden=256, Batch=64
    {'input_dim':784, 'hidden_dim': 256, 'lr': 0.01, 'batch_size': 64,  'epochs': 20},
    # Hidden=256, Batch=128
    {'input_dim':784, 'hidden_dim': 256, 'lr': 0.01, 'batch_size': 128, 'epochs': 20},
    # Hidden=256, Batch=256
    {'input_dim':784, 'hidden_dim': 256, 'lr': 0.01, 'batch_size': 256, 'epochs': 20},
]

print(f"✓ Grid defined with {len(experiments)} configurations")

# ============================================================================
# CELL 5: Run the grid search
# ============================================================================

results = []
print(f"\nStarting Grid Search on {len(experiments)} configurations...\n")

for i, conf in enumerate(experiments):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(experiments)}")
    print(f"Config: Hidden={conf['hidden_dim']}, LR={conf['lr']}, Batch={conf['batch_size']}")
    print(f"{'='*60}")
    
    # Run the experiment using loaded datasets
    loss, auc = run_experiment(conf, train_ds, val_ds, test_ds)

    # Store results
    res = conf.copy()
    res['val_loss'] = loss
    res['val_auc'] = auc
    results.append(res)

    print(f"\n✓ Exp {i+1} Complete: Val Loss={loss:.4f}, Val AUC={auc:.4f}")

# ============================================================================
# CELL 6: Display results
# ============================================================================

# Display Leaderboard
df_results = pd.DataFrame(results)
print(f"\n{'='*70}")
print(f" FINAL LEADERBOARD (Sorted by Best Val AUC)")
print(f"{'='*70}")
print(df_results.sort_values(by='val_auc', ascending=False).to_string(index=False))
print(f"{'='*70}")

# Find best configuration
best_idx = df_results['val_auc'].idxmax()
best_config = df_results.loc[best_idx]

print(f"\n🏆 BEST CONFIGURATION:")
print(f"   Hidden Dim: {best_config['hidden_dim']}")
print(f"   Learning Rate: {best_config['lr']}")
print(f"   Batch Size: {best_config['batch_size']}")
print(f"   Val AUC: {best_config['val_auc']:.4f}")
print(f"   Val Loss: {best_config['val_loss']:.4f}")
