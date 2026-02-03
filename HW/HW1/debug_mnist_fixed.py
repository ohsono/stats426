import numpy as np
import csv
import matplotlib.pyplot as plt

def _load_mnist_dataset(path=None):
    with open(path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        # Skip the header row
        next(csvreader, None)
        for row in csvreader:
            yield row

def plot_data(path):      
    images = []
    labels = []
    
    # Load first 64 images for 8x8 grid
    for data in _load_mnist_dataset(path):
        if len(images) >= 64:
            break
            
        label = data[0]
        # Pixel data is normalized (0-1) in this dataset based on previous inspection
        pixels = np.array(data[1:], dtype='float32').reshape((28, 28))
        
        images.append(pixels)
        labels.append(label)

    # created 8x8 grid
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle('MNIST 8x8 Grid', fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f"Label: {labels[i]}", fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off') # Hide unused subplots
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_data('./mnist_test.csv')
