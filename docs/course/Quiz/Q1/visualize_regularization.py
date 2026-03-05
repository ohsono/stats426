import numpy as np
import matplotlib.pyplot as plt

def plot_regularization_intuition():
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Grid range
    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x, y)
    
    # 1. Lasso (L1) = Diamond Constraint
    # Constraint: |w1| + |w2| <= 1
    l1_constraint = np.abs(X) + np.abs(Y)
    
    # Loss function (MSE) - Elliptical contours centered at (1.5, 1.5)
    # Loss = (w1 - 1.5)^2 + (w2 - 1.5)^2
    # We want to minimize this subject to constraint
    loss = (X - 1.5)**2 + (Y - 1.5)**2
    
    # Plot L1 Constraint Region
    ax1.contourf(X, Y, l1_constraint, levels=[-np.inf, 1], colors=['#e0f7fa'], alpha=0.5)
    ax1.contour(X, Y, l1_constraint, levels=[1], colors=['#006064'], linewidths=2)
    
    # Plot Loss Contours for L1
    # We choose levels that will "touch" the diamond
    ax1.contour(X, Y, loss, levels=[1.125, 2, 3, 5, 8], colors=['#ff6f00'], linewidths=1.5)
    
    # Intersection point for Lasso (Corner)
    ax1.plot(1, 0, 'ro', markersize=8, label='Optimal Solution')
    ax1.text(1.1, 0.1, 'Touches at Corner\n(w2 = 0 → Sparsity)', fontsize=10, fontweight='bold')
    
    ax1.set_title("Lasso (L1) Regularization\nGeometric Intuition", fontsize=14)
    ax1.set_xlim(-1.5, 2.5)
    ax1.set_ylim(-1.5, 2.5)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('Weight 1 (w1)')
    ax1.set_ylabel('Weight 2 (w2)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # 2. Ridge (L2) = Circle Constraint
    # Constraint: w1^2 + w2^2 <= 1
    l2_constraint = X**2 + Y**2
    
    # Plot L2 Constraint Region
    ax2.contourf(X, Y, l2_constraint, levels=[-np.inf, 1], colors=['#e8f5e9'], alpha=0.5)
    ax2.contour(X, Y, l2_constraint, levels=[1], colors=['#1b5e20'], linewidths=2)
    
    # Plot Loss Contours for L2
    # We choose levels that will "touch" the circle
    ax2.contour(X, Y, loss, levels=[0.75, 2, 3, 5, 8], colors=['#ff6f00'], linewidths=1.5)
    
    # Intersection point for Ridge (Smooth edge)
    # The closest point on unit circle to (1.5, 1.5) is (1/sqrt(2), 1/sqrt(2))
    pt = 1 / np.sqrt(2)
    ax2.plot(pt, pt, 'ro', markersize=8, label='Optimal Solution')
    ax2.text(pt + 0.1, pt, 'Touches at Edge\n(Both w1, w2 ≠ 0)', fontsize=10, fontweight='bold')
    
    ax2.set_title("Ridge (L2) Regularization\nGeometric Intuition", fontsize=14)
    ax2.set_xlim(-1.5, 2.5)
    ax2.set_ylim(-1.5, 2.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Weight 1 (w1)')
    ax2.set_ylabel('Weight 2 (w2)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('l1_vs_l2_geometry.png', dpi=150)
    print("Image generated: l1_vs_l2_geometry.png")

if __name__ == "__main__":
    plot_regularization_intuition()
