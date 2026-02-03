import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

class SingleMLP(nn.Module):
    """
    Single-layer MLP for binary classification with integrated training and evaluation methods.
    """
    
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        super(SingleMLP, self).__init__()
        # Input Layer -> Hidden Layer
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # Hidden Layer -> Output (Scalar)
        self.output = nn.Linear(hidden_dim, 1)  # Binary output
        # Activation function
        self.activation = nn.ReLU()
        
        # Store hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Proper Initialization
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
        # Training history
        self.train_loss_history = []
        self.val_loss_history = []

    @staticmethod
    def load_filter_data(path, name="Dataset", target_digits=None):
        """
        Loads the dataset from a CSV file, filters for specified digits, and prepares tensors.

        Args:
            path (str): The file path to the CSV dataset.
            name (str, optional): A name for the dataset for logging purposes.
            target_digits (list, optional): List of digits to filter. Defaults to [3, 5].

        Returns:
            tuple: (X, y) torch.Tensors
        """
        print(f"Loading {name} from {path}...")
        df = pd.read_csv(path)
    
        label_col = df.columns[0]
        
        # Default to digits 3 and 5 for binary classification
        if target_digits is None:
            target_digits = [3, 5]
        
        df_filtered = df[df[label_col].isin(target_digits)].copy()

        if df_filtered.empty:
            raise ValueError(f"Error: No data found for digits {target_digits} in {path}.")
            
        # Map 3 -> 0, 5 -> 1 (for binary classification)
        y = df_filtered[label_col].apply(lambda x: 0 if x == target_digits[0] else 1).values
        
        # Features. Drop label column.
        X = df_filtered.drop(columns=[label_col]).values
        
        # Ensure float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        print(f"  Shape after filtering: {X.shape}")
        
        return torch.tensor(X), torch.tensor(y).unsqueeze(1)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.activation(self.layer1(x))
        return self.output(x)
    
    def train_model(self, train_loader, val_loader, epochs, criterion=None, optimizer=None, verbose=True):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            criterion: Loss function (defaults to BCEWithLogitsLoss)
            optimizer: Optimizer (defaults to SGD with self.lr)
            verbose: Whether to print progress
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        
        self.train_loss_history = []
        self.val_loss_history = []
        
        if verbose:
            print("\nStarting Training...")
        
        for epoch in range(epochs):
            # Training Phase
            self.train()
            running_train_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
            
            avg_train_loss = running_train_loss / len(train_loader.dataset)
            self.train_loss_history.append(avg_train_loss)
            
            # Validation Phase
            self.eval()
            running_val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
            
            avg_val_loss = running_val_loss / len(val_loader.dataset)
            self.val_loss_history.append(avg_val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            dict: Dictionary containing targets, predictions, accuracy, and AUC
        """
        self.eval()
        targets = []
        preds = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                probs = torch.sigmoid(outputs)
                targets.extend(labels.numpy())
                preds.extend(probs.numpy())
        
        targets = np.array(targets)
        preds = np.array(preds)
        
        acc = accuracy_score(targets, (preds > 0.5).astype(int))
        auc = roc_auc_score(targets, preds)
        
        return {
            'targets': targets,
            'predictions': preds,
            'accuracy': acc,
            'auc': auc
        }
    
    def plot_training_history(self, save_path='training_loss.png'):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.title('Training Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved loss plot to {save_path}")
    
    def plot_roc_curve(self, targets, predictions, save_path='roc_curve.png'):
        """
        Plot ROC curve.
        
        Args:
            targets: True labels
            predictions: Predicted probabilities
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(targets, predictions)
        auc = roc_auc_score(targets, predictions)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved ROC curve to {save_path}")


def run_experiment(config, train_ds, val_ds, test_ds):
    """
    Run a single experiment with given configuration.
    
    Args:
        config: Dictionary with 'input_dim', 'hidden_dim', 'lr', 'batch_size', 'epochs'
        train_ds: Training TensorDataset
        val_ds: Validation TensorDataset
        test_ds: Test TensorDataset
        
    Returns:
        tuple: (val_loss, val_auc)
    """
    # Unpack configuration
    i_dim = config['input_dim']
    h_dim = config['hidden_dim']
    lr = config['lr']
    bs = config['batch_size']
    epochs = config['epochs']
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    # Initialize model
    model = SingleMLP(input_dim=i_dim, hidden_dim=h_dim, lr=lr)
    
    # Train model
    model.train_model(train_loader, val_loader, epochs, verbose=False)
    
    # Evaluate on validation set
    val_results = model.evaluate(val_loader)
    
    print(f"✓ Val Accuracy: {val_results['accuracy']:.4f}, Val AUC: {val_results['auc']:.4f}")
    
    # Plot results
    model.plot_training_history('mnist_experiment_loss.png')
    model.plot_roc_curve(val_results['targets'], val_results['predictions'], 'mnist_roc_curve.png')
    
    return val_results['auc'], model


if __name__ == "__main__":
    # Paths
    train_path = './mnist_train.csv'
    val_path = './mnist_val.csv'
    test_path = './mnist_test.csv'
    
    # Load Data
    X_train, y_train = SingleMLP.load_filter_data(train_path, "Train")
    X_val, y_val = SingleMLP.load_filter_data(val_path, "Val")
    X_test, y_test = SingleMLP.load_filter_data(test_path, "Test")
    
    # Create Datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    # Grid Search Configuration
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
    
    print(f"Grid defined with {len(experiments)} configurations.\n")
    
    # Run Grid Search
    results = []
    best_models = []
    
    for i, conf in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}")
        print(f"Config: Hidden={conf['hidden_dim']}, Batch={conf['batch_size']}")
        print(f"{'='*60}")
        
        auc, model = run_experiment(conf, train_ds, val_ds, test_ds)
        
        res = conf.copy()
        res['val_auc'] = auc
        results.append(res)
        best_models.append(model)
    
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
    print(f"   Batch Size: {best_config['batch_size']}")
    print(f"   Val AUC: {best_config['val_auc']:.4f}")
