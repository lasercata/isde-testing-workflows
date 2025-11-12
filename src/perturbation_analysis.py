import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import load_and_split_mnist
from src.data_perturb.CDataPerturbRandom import CDataPerturbRandom
from src.data_perturb.CDataPerturbGaussian import CDataPerturbGaussian


class NMCClassifier:
    """Nearest Mean Centroid Classifier"""
    
    def __init__(self):
        self.centroids = None
        self.classes = None
    
    def fit(self, X_train, y_train):
        """Train the NMC classifier by computing class centroids"""
        self.classes = np.unique(y_train)
        self.centroids = np.zeros((len(self.classes), X_train.shape[1]))
        
        for i, c in enumerate(self.classes):
            self.centroids[i] = X_train[y_train == c].mean(axis=0)
        
        return self
    
    def predict(self, X_test):
        """Predict class labels by finding nearest centroid"""
        predictions = np.zeros(X_test.shape[0], dtype=int)
        
        for i, x in enumerate(X_test):
            # Compute distances to all centroids
            distances = np.linalg.norm(self.centroids - x, axis=1)
            # Predict the class of the nearest centroid
            predictions[i] = self.classes[np.argmin(distances)]
        
        return predictions
    
    def score(self, X_test, y_test):
        """Compute classification accuracy"""
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy


def plot_ten_images(images, labels, title="Images", predictions=None):
    """Plot 10 images in a 2x5 grid"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Reshape to 28x28 for visualization
            img = images[i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            
            if predictions is not None:
                ax.set_title(f'True: {labels[i]}\nPred: {predictions[i]}')
            else:
                ax.set_title(f'Label: {labels[i]}')
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def test_perturbations_visual():
    """Test both perturbation models on 10 random images and visualize results"""
    print("=" * 60)
    print("TASK 1: Visual comparison of perturbation models")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_mnist()
    
    # Select 10 random images from test set
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test), 10, replace=False)
    sample_images = X_test[random_indices]
    sample_labels = y_test[random_indices]
    
    # Original images
    fig1 = plot_ten_images(sample_images, sample_labels, title="Original Images")
    plt.savefig('results_original_images.png', dpi=100, bbox_inches='tight')
    print("✓ Saved original images to results_original_images.png")
    
    # Perturb with Random model (K=100)
    perturb_random = CDataPerturbRandom(K=100)
    sample_images_random = sample_images.copy()
    perturbed_random = perturb_random.perturb_dataset(sample_images_random)
    
    fig2 = plot_ten_images(perturbed_random, sample_labels, 
                          title="Random Perturbation (K=100)")
    plt.savefig('results_random_perturbation.png', dpi=100, bbox_inches='tight')
    print("✓ Saved random perturbation to results_random_perturbation.png")
    
    # Perturb with Gaussian model (sigma=100)
    perturb_gaussian = CDataPerturbGaussian(sig=100.0)
    sample_images_gaussian = sample_images.copy()
    perturbed_gaussian = perturb_gaussian.perturb_dataset(sample_images_gaussian)
    
    fig3 = plot_ten_images(perturbed_gaussian, sample_labels, 
                          title="Gaussian Perturbation (sigma=100)")
    plt.savefig('results_gaussian_perturbation.png', dpi=100, bbox_inches='tight')
    print("✓ Saved gaussian perturbation to results_gaussian_perturbation.png")
    
    plt.show()


def train_and_evaluate_nmc():
    """Train NMC classifier and evaluate on clean test set"""
    print("\n" + "=" * 60)
    print("TASK 2: Train NMC classifier and evaluate on clean test set")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_mnist()
    
    # Train NMC classifier
    print("Training NMC classifier...")
    nmc = NMCClassifier()
    nmc.fit(X_train, y_train)
    
    # Evaluate on clean test set
    accuracy = nmc.score(X_test, y_test)
    print(f"✓ Classification accuracy on clean test set: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return nmc, X_train, X_test, y_train, y_test


def evaluate_perturbations(nmc, X_test, y_test):
    """Evaluate classifier with different perturbation parameters"""
    print("\n" + "=" * 60)
    print("TASK 3: Evaluate perturbations with different parameters")
    print("=" * 60)
    
    # Parameters to test
    K_values = [0, 10, 20, 50, 100, 200, 500]
    sigma_values = [10, 20, 50, 200, 500]
    
    # Storage for results
    accuracies_K = []
    accuracies_sigma = []
    
    # Test Random perturbation with different K values
    print("\nTesting Random Perturbation with K =", K_values)
    for K in K_values:
        perturb = CDataPerturbRandom(K=K)
        X_test_perturbed = perturb.perturb_dataset(X_test.copy())
        accuracy = nmc.score(X_test_perturbed, y_test)
        accuracies_K.append(accuracy)
        print(f"  K={K:3d}: accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Test Gaussian perturbation with different sigma values
    print(f"\nTesting Gaussian Perturbation with sigma = {sigma_values}")
    for sigma in sigma_values:
        perturb = CDataPerturbGaussian(sig=sigma)
        X_test_perturbed = perturb.perturb_dataset(X_test.copy())
        accuracy = nmc.score(X_test_perturbed, y_test)
        accuracies_sigma.append(accuracy)
        print(f"  sigma={sigma:3d}: accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return K_values, accuracies_K, sigma_values, accuracies_sigma


def plot_accuracy_results(K_values, accuracies_K, sigma_values, accuracies_sigma):
    """Create plots showing accuracy vs perturbation parameters"""
    print("\n" + "=" * 60)
    print("TASK 4: Plotting accuracy vs perturbation parameters")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy vs K
    ax1.plot(K_values, accuracies_K, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('K (number of randomly perturbed pixels)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs K (Random Perturbation)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add percentage labels on points
    for k, acc in zip(K_values, accuracies_K):
        ax1.annotate(f'{acc*100:.1f}%', 
                    (k, acc), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # Plot accuracy vs sigma
    ax2.plot(sigma_values, accuracies_sigma, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Sigma (Gaussian noise standard deviation)', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Sigma (Gaussian Perturbation)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Add percentage labels on points
    for sig, acc in zip(sigma_values, accuracies_sigma):
        ax2.annotate(f'{acc*100:.1f}%', 
                    (sig, acc), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_accuracy_vs_perturbation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved accuracy plots to results_accuracy_vs_perturbation.png")
    plt.show()


def main():
    """Main function to run all analysis tasks"""
    print("\n" + "=" * 60)
    print("MNIST PERTURBATION ANALYSIS")
    print("=" * 60)
    
    # Task 1: Visual comparison of perturbations
    test_perturbations_visual()
    
    # Task 2: Train NMC and evaluate on clean data
    nmc, X_train, X_test, y_train, y_test = train_and_evaluate_nmc()
    
    # Task 3: Evaluate with different perturbation parameters
    K_values, accuracies_K, sigma_values, accuracies_sigma = evaluate_perturbations(
        nmc, X_test, y_test
    )
    
    # Task 4: Plot results
    plot_accuracy_results(K_values, accuracies_K, sigma_values, accuracies_sigma)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results_original_images.png")
    print("  - results_random_perturbation.png")
    print("  - results_gaussian_perturbation.png")
    print("  - results_accuracy_vs_perturbation.png")
    print()


if __name__ == "__main__":
    main()
