# Grid Search Configuration - Summary

## Assignment Requirements
Your assignment specifies:
- **Mini-batch size s ∈ {64, 128, 256}**
- **Hidden layer dimension h ∈ {64, 128, 256}**

This creates a **3 × 3 = 9 total configurations** to test.

## What Changed

### Before (Incorrect - 5 custom configurations):
```python
experiments = [
    {'input_dim':784, 'hidden_dim': 32,  'lr': 0.01, 'batch_size': 64,  'epochs': 30},
    {'input_dim':784, 'hidden_dim': 128, 'lr': 0.1,  'batch_size': 64,  'epochs': 30},
    {'input_dim':784, 'hidden_dim': 128, 'lr': 0.01, 'batch_size': 32,  'epochs': 30},
    {'input_dim':784, 'hidden_dim': 512, 'lr': 0.1,  'batch_size': 256, 'epochs': 30},
    {'input_dim':784, 'hidden_dim': 32,  'lr': 0.5,  'batch_size': 64,  'epochs': 30},
]
```

### After (Correct - 9 grid search configurations):
```python
hidden_dims = [64, 128, 256]
batch_sizes = [64, 128, 256]
experiments = []

for h in hidden_dims:
    for b in batch_sizes:
        experiments.append({
            'input_dim': 784,
            'hidden_dim': h,
            'lr': 0.01,      # Fixed learning rate
            'batch_size': b,
            'epochs': 20     # Fixed epochs
        })
```

## The 9 Configurations

| Config | Hidden Dim | Batch Size | Learning Rate | Epochs |
|--------|-----------|------------|---------------|--------|
| 1      | 64        | 64         | 0.01          | 20     |
| 2      | 64        | 128        | 0.01          | 20     |
| 3      | 64        | 256        | 0.01          | 20     |
| 4      | 128       | 64         | 0.01          | 20     |
| 5      | 128       | 128        | 0.01          | 20     |
| 6      | 128       | 256        | 0.01          | 20     |
| 7      | 256       | 64         | 0.01          | 20     |
| 8      | 256       | 128        | 0.01          | 20     |
| 9      | 256       | 256        | 0.01          | 20     |

## Fixed Parameters
- **Input dimension**: 784 (28×28 MNIST images)
- **Learning rate**: 0.01 (fixed across all experiments)
- **Epochs**: 20 (fixed across all experiments)

## Files Updated
1. ✅ `mnist_binary_classification.py` - Standalone script
2. ✅ `notebook_corrected_code.py` - Reference code for your notebook

## Next Steps
The script is now running with the correct grid search. It will:
1. Train 9 different model configurations
2. Evaluate each on the validation set
3. Report validation loss and AUC for each
4. Display a leaderboard sorted by validation AUC
5. Identify the best configuration
