#!/usr/bin/env python3
"""Script to visualize transfer learning results"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

# Load CIFAR-10 to get class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_training_history():
    """
    NOTE: Cette fonction nécessite de sauvegarder l'historique pendant l'entraînement.
    Pour l'instant, on va créer des exemples de visualisation avec le modèle entraîné.
    """
    print("Pour tracer l'historique d'entraînement, il faudrait sauvegarder history.history")
    print("pendant l'entraînement. Créons d'autres visualisations à la place...\n")

def plot_sample_predictions():
    """Show sample predictions from the model"""
    print("=== Visualizing Sample Predictions ===")
    
    # Load model
    model = K.models.load_model('cifar10.h5')
    
    # Load test data
    _, (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    # Preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    X_test_resized = tf.image.resize(X_test[:20], (96, 96))
    X_test_preprocessed = preprocess_input(X_test_resized)
    
    # Predict
    predictions = model.predict(X_test_preprocessed, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Sample Predictions from Transfer Learning Model', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        # Show original image (32x32)
        ax.imshow(X_test[idx])
        
        true_label = Y_test[idx][0]
        pred_label = predicted_classes[idx]
        confidence = predictions[idx][pred_label] * 100
        
        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        ax.set_title(f'True: {class_names[true_label]}\n'
                    f'Pred: {class_names[pred_label]} ({confidence:.1f}%)',
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: sample_predictions.png\n")
    plt.show()

def plot_confusion_matrix():
    """Plot confusion matrix"""
    print("=== Creating Confusion Matrix ===")
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Load model
    model = K.models.load_model('cifar10.h5')
    
    # Load test data
    _, (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    # Preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("Preprocessing test data...")
    X_test_resized = tf.image.resize(X_test, (96, 96))
    X_test_preprocessed = preprocess_input(X_test_resized)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(X_test_preprocessed, batch_size=128, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(Y_test, predicted_classes)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Transfer Learning on CIFAR-10', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: confusion_matrix.png\n")
    plt.show()

def plot_per_class_accuracy():
    """Plot accuracy per class"""
    print("=== Calculating Per-Class Accuracy ===")
    
    # Load model
    model = K.models.load_model('cifar10.h5')
    
    # Load test data
    _, (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    # Preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("Preprocessing test data...")
    X_test_resized = tf.image.resize(X_test, (96, 96))
    X_test_preprocessed = preprocess_input(X_test_resized)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(X_test_preprocessed, batch_size=128, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate per-class accuracy
    accuracies = []
    for i in range(10):
        mask = (Y_test.flatten() == i)
        class_acc = np.mean(predicted_classes[mask] == i)
        accuracies.append(class_acc * 100)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies, color='steelblue', edgecolor='black')
    
    # Color bars: green if >85%, yellow if 75-85%, red if <75%
    for bar, acc in zip(bars, accuracies):
        if acc >= 85:
            bar.set_color('green')
        elif acc >= 75:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=87, color='r', linestyle='--', label='Target: 87%')
    plt.title('Per-Class Accuracy - Transfer Learning Model', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 100])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: per_class_accuracy.png\n")
    plt.show()

def plot_model_architecture():
    """Visualize model architecture"""
    print("=== Model Architecture Summary ===")
    
    model = K.models.load_model('cifar10.h5')
    
    # Print summary
    print("\nModel Summary:")
    model.summary()
    
    # Try to plot model (requires graphviz and pydot)
    try:
        K.utils.plot_model(
            model, 
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )
        print("\n✓ Saved: model_architecture.png")
    except Exception as e:
        print(f"\n⚠ Could not plot model architecture: {e}")
        print("Install graphviz and pydot for model visualization:")
        print("  sudo apt-get install graphviz")
        print("  pip install pydot")

def plot_before_after_comparison():
    """Show before/after preprocessing"""
    print("=== Before/After Preprocessing Comparison ===")
    
    # Load original data
    (X_train, _), _ = K.datasets.cifar10.load_data()
    
    # Take 8 samples
    samples = X_train[:8]
    
    # Preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    samples_resized = tf.image.resize(samples, (96, 96))
    samples_preprocessed = preprocess_input(samples_resized)
    
    # Denormalize for visualization
    samples_denorm = (samples_preprocessed + 1) / 2  # Back to [0, 1]
    
    # Plot
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('Before vs After Preprocessing (32x32 → 96x96 + MobileNetV2 preprocessing)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(8):
        # Original
        axes[0, i].imshow(samples[i])
        axes[0, i].set_title(f'Original\n32x32', fontsize=9)
        axes[0, i].axis('off')
        
        # Preprocessed
        axes[1, i].imshow(samples_denorm[i])
        axes[1, i].set_title(f'Preprocessed\n96x96', fontsize=9)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('before_after_preprocessing.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: before_after_preprocessing.png\n")
    plt.show()

if __name__ == '__main__':
    print("🎨 Generating Visualizations for Transfer Learning Project\n")
    print("=" * 60)
    
    # 1. Sample predictions
    plot_sample_predictions()
    
    # 2. Before/after preprocessing
    plot_before_after_comparison()
    
    # 3. Per-class accuracy
    plot_per_class_accuracy()
    
    # 4. Confusion matrix
    plot_confusion_matrix()
    
    # 5. Model architecture
    plot_model_architecture()
    
    print("=" * 60)
    print("\n✅ All visualizations generated!")
    print("\nGenerated files:")
    print("  - sample_predictions.png")
    print("  - before_after_preprocessing.png")
    print("  - per_class_accuracy.png")
    print("  - confusion_matrix.png")
    print("  - model_architecture.png (if dependencies available)")