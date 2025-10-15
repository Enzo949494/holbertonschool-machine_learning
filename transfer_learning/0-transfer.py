#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model
    
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    
    Returns:
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    # Resize images from 32x32 to 96x96 (MobileNetV2 minimum)
    X_p = tf.image.resize(X, (96, 96))
    
    # Apply MobileNetV2 specific preprocessing
    X_p = preprocess_input(X_p)
    
    # One-hot encode the labels
    Y_p = K.utils.to_categorical(Y, 10)
    
    return X_p.numpy(), Y_p


def preprocess_data_batched(X, Y, batch_size=500):
    """
    Pre-processes the data in batches to avoid OOM
    """
    num_samples = len(X)
    X_p_list = []
    
    print(f"Processing {num_samples} images in batches of {batch_size}...")
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_X = X[i:end_idx]
        
        # Resize batch to 96x96
        batch_X_resized = tf.image.resize(batch_X, (96, 96))
        batch_X_preprocessed = preprocess_input(batch_X_resized)
        
        X_p_list.append(batch_X_preprocessed.numpy())
        
        print(f"  Processed {end_idx}/{num_samples} images...")
    
    X_p = np.concatenate(X_p_list, axis=0)
    Y_p = K.utils.to_categorical(Y, 10)
    
    return X_p, Y_p


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    print("Preprocessing data in batches to avoid OOM...")
    X_train_p, Y_train_p = preprocess_data_batched(X_train, Y_train, batch_size=500)
    X_test_p, Y_test_p = preprocess_data_batched(X_test, Y_test, batch_size=500)
    
    # Create validation set from training data
    validation_split = 0.2
    split_idx = int(len(X_train_p) * (1 - validation_split))
    X_val = X_train_p[split_idx:]
    Y_val = Y_train_p[split_idx:]
    X_train_p = X_train_p[:split_idx]
    Y_train_p = Y_train_p[:split_idx]
    
    print(f"\nTraining samples: {len(X_train_p)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test_p)}")
    
    # ========== PHASE 1: FEATURE EXTRACTION ==========
    print("\n=== Phase 1: Feature Extraction ===")
    
    # Load base model - MobileNetV2 (lightweight!)
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(96, 96, 3),
        pooling='avg'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    print("Extracting features from frozen base model...")
    # Extract features ONCE (Hint 3) - in batches
    train_features = base_model.predict(X_train_p, batch_size=64, verbose=1)
    val_features = base_model.predict(X_val, batch_size=64, verbose=1)
    test_features = base_model.predict(X_test_p, batch_size=64, verbose=1)
    
    print(f"Feature shape: {train_features.shape}")
    
    # Free up memory
    del X_train_p, X_val, X_test_p
    import gc
    gc.collect()
    
    # Build classifier on top of features
    inputs = K.Input(shape=train_features.shape[1:])
    x = K.layers.Dropout(0.5)(inputs)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    
    classifier = K.Model(inputs=inputs, outputs=outputs)
    
    # Compile classifier
    classifier.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining classifier on extracted features...")
    # Train classifier on features
    history1 = classifier.fit(
        train_features, Y_train_p,
        validation_data=(val_features, Y_val),
        epochs=30,
        batch_size=128,
        verbose=1,
        callbacks=[
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True
            )
        ]
    )
    
    # ========== PHASE 2: FINE-TUNING ==========
    print("\n=== Phase 2: Fine-Tuning ===")
    
    # Reload preprocessed data for fine-tuning
    print("Reloading preprocessed data for fine-tuning...")
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = K.datasets.cifar10.load_data()
    # FIX: Récupérer AUSSI les labels Y_train_p
    X_train_p, Y_train_p_reload = preprocess_data_batched(X_train_orig, Y_train_orig, batch_size=500)
    X_test_p, Y_test_p = preprocess_data_batched(X_test_orig, Y_test_orig, batch_size=500)
    
    # Recreate validation split
    X_val = X_train_p[split_idx:]
    Y_val = Y_train_p_reload[split_idx:]  # FIX: Utiliser les nouveaux labels
    X_train_p = X_train_p[:split_idx]
    Y_train_p = Y_train_p_reload[:split_idx]  # FIX: Utiliser les nouveaux labels
    
    # Build full model for fine-tuning
    base_model_full = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(96, 96, 3),
        pooling='avg'
    )
    
    # Unfreeze only the last layers
    base_model_full.trainable = True
    for layer in base_model_full.layers[:-20]:
        layer.trainable = False
    
    # Build complete model
    inputs = K.Input(shape=(96, 96, 3))
    x = base_model_full(inputs, training=False)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    
    model = K.Model(inputs=inputs, outputs=outputs)
    
    # Transfer weights from trained classifier
    model.layers[-3].set_weights(classifier.layers[2].get_weights())
    model.layers[-1].set_weights(classifier.layers[4].get_weights())
    
    # Compile with very low learning rate
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    
    print("\nFine-tuning the model...")
    # Fine-tune
    history2 = model.fit(
        datagen.flow(X_train_p, Y_train_p, batch_size=64),
        validation_data=(X_val, Y_val),
        epochs=50,
        steps_per_epoch=len(X_train_p) // 64,
        verbose=1,
        callbacks=[
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
    )
    
    # Evaluate on test set
    print("\n=== Final Evaluation ===")
    test_loss, test_acc = model.evaluate(X_test_p, Y_test_p, batch_size=128, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    if test_acc >= 0.87:
        print(f"\n✓ Target accuracy reached! Saving model...")
        model.save('cifar10.h5')
        print("Model saved as cifar10.h5")
    else:
        print(f"\n✗ Target accuracy not reached (current: {test_acc:.4f}, target: 0.87)")
        print("Saving model anyway for inspection...")
        model.save('cifar10.h5')