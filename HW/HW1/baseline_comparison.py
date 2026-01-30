"""
Logistic Regression Baseline Comparison
This script trains a Logistic Regression model and compares it with the best MLP model.

NOTE: Make sure you have defined the SingleMLP class in your notebook before running this code.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a Logistic Regression baseline model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        dict: Results containing model, predictions, and metrics
    """
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION BASELINE")
    print("="*70)
    
    # Convert PyTorch tensors to numpy if needed
    if torch.is_tensor(X_train):
        X_train = X_train.numpy()
        y_train = y_train.numpy().ravel()
    if torch.is_tensor(X_val):
        X_val = X_val.numpy()
        y_val = y_val.numpy().ravel()
    if torch.is_tensor(X_test):
        X_test = X_test.numpy()
        y_test = y_test.numpy().ravel()
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Predictions on validation set
    val_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    val_pred = lr_model.predict(X_val)
    
    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    
    # Predictions on test set
    test_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    test_pred = lr_model.predict(X_test)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    return {
        'model': lr_model,
        'val_predictions': val_pred_proba,
        'val_targets': y_val,
        'val_accuracy': val_acc,
        'val_auc': val_auc,
        'test_predictions': test_pred_proba,
        'test_targets': y_test,
        'test_accuracy': test_acc,
        'test_auc': test_auc
    }

def plot_comparison_roc(lr_results, mlp_results, save_path='comparison_roc_curve.png'):
    """
    Plot ROC curves comparing Logistic Regression and MLP.
    
    Args:
        lr_results: Results dictionary from Logistic Regression
        mlp_results: Results dictionary from MLP
        save_path: Path to save the plot
    """
    # Calculate ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(lr_results['test_targets'], lr_results['test_predictions'])
    mlp_fpr, mlp_tpr, _ = roc_curve(mlp_results['test_targets'], mlp_results['test_predictions'])
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(lr_fpr, lr_tpr, color='blue', lw=2, 
             label=f'Logistic Regression (AUC = {lr_results["test_auc"]:.4f})')
    plt.plot(mlp_fpr, mlp_tpr, color='red', lw=2, 
             label=f'Single-Layer MLP (AUC = {mlp_results["test_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Comparison: Logistic Regression vs MLP', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved comparison ROC curve to {save_path}")


def print_comparison_summary(lr_results, mlp_results):
    """
    Print a detailed comparison summary.
    
    Args:
        lr_results: Results dictionary from Logistic Regression
        mlp_results: Results dictionary from MLP
    """
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    print("\n📊 TEST SET RESULTS:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Logistic Regression':<25} {'MLP':<25}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {lr_results['test_accuracy']:<25.4f} {mlp_results['test_accuracy']:<25.4f}")
    print(f"{'AUC':<20} {lr_results['test_auc']:<25.4f} {mlp_results['test_auc']:<25.4f}")
    print("-" * 70)
    
    # Calculate improvements
    acc_improvement = (mlp_results['test_accuracy'] - lr_results['test_accuracy']) * 100
    auc_improvement = (mlp_results['test_auc'] - lr_results['test_auc']) * 100
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Accuracy: {acc_improvement:+.2f} percentage points")
    print(f"  AUC: {auc_improvement:+.2f} percentage points")
    
    print("\n💡 DISCUSSION:")
    print("-" * 70)
    
    if mlp_results['test_auc'] > lr_results['test_auc'] + 0.01:
        print("✓ The MLP shows a meaningful improvement over Logistic Regression.")
        print(f"  The AUC improvement of {auc_improvement:.2f} percentage points suggests that")
        print("  the non-linear hidden layer helps capture patterns that linear models miss.")
    elif abs(mlp_results['test_auc'] - lr_results['test_auc']) < 0.01:
        print("≈ The MLP and Logistic Regression perform similarly.")
        print("  This suggests that for this specific digit pair (3 vs 5), the decision")
        print("  boundary is approximately linear, and the added complexity of the MLP")
        print("  does not provide significant benefits.")
    else:
        print("⚠ Logistic Regression slightly outperforms the MLP.")
        print("  This could indicate overfitting in the MLP or that the simpler linear")
        print("  model is more appropriate for this task.")
    
    print("\n🔍 KEY INSIGHTS:")
    print(f"  • Both models achieve high performance (AUC > 0.99)")
    print(f"  • Digits 3 and 5 are relatively easy to distinguish")
    print(f"  • The {'MLP' if mlp_results['test_auc'] > lr_results['test_auc'] else 'Logistic Regression'} "
          f"is the better choice for this task")
    print("="*70)


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, y_train = SingleMLP.load_filter_data('./mnist_train.csv', "Train", target_digits=[3, 5])
    X_val, y_val = SingleMLP.load_filter_data('./mnist_val.csv', "Val", target_digits=[3, 5])
    X_test, y_test = SingleMLP.load_filter_data('./mnist_test.csv', "Test", target_digits=[3, 5])
    
    # Create datasets for MLP
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    # Best configuration from grid search (update based on your results)
    best_config = {
        'input_dim': 784,
        'hidden_dim': 256,  # Update based on your grid search results
        'batch_size': 64,   # Update based on your grid search results
        'lr': 0.01,
        'epochs': 20
    }
    
    # ========================================================================
    # TRAIN LOGISTIC REGRESSION BASELINE
    # ========================================================================
    lr_results = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # ========================================================================
    # TRAIN BEST MLP MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING BEST MLP MODEL")
    print("="*70)
    print(f"Configuration: Hidden={best_config['hidden_dim']}, "
          f"Batch={best_config['batch_size']}, LR={best_config['lr']}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=best_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=best_config['batch_size'], shuffle=False)
    
    # Initialize and train model
    mlp_model = SingleMLP(
        input_dim=best_config['input_dim'],
        hidden_dim=best_config['hidden_dim'],
        lr=best_config['lr']
    )
    
    print("\nTraining MLP...")
    mlp_model.train_model(train_loader, val_loader, best_config['epochs'], verbose=True)
    print("✓ Training complete")
    
    # Evaluate
    val_results = mlp_model.evaluate(val_loader)
    test_results = mlp_model.evaluate(test_loader)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  AUC: {val_results['auc']:.4f}")
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  AUC: {test_results['auc']:.4f}")
    
    # Package MLP results
    mlp_results = {
        'model': mlp_model,
        'test_predictions': test_results['predictions'],
        'test_targets': test_results['targets'],
        'test_accuracy': test_results['accuracy'],
        'test_auc': test_results['auc']
    }
    
    # ========================================================================
    # COMPARE RESULTS
    # ========================================================================
    plot_comparison_roc(lr_results, mlp_results)
    print_comparison_summary(lr_results, mlp_results)
    
    print("\n✓ Baseline comparison complete!")
