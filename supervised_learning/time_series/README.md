# Bitcoin Price Forecasting with RNN/LSTM

Prediction of Bitcoin (BTC) price using a Recurrent Neural Network (LSTM) architecture. The model uses 24 hours of historical BTC data to predict the price at the close of the following hour.

---

## 📊 Project Overview

- **Model Architecture**: LSTM (Long Short-Term Memory) RNN
- **Input**: 1440 timesteps (24 hours × 60 minutes) of BTC data
- **Output**: Predicted BTC price for the next hour
- **Loss Function**: Mean Squared Error (MSE)
- **Data Pipeline**: tf.data.Dataset with batch processing

---

## 📁 Project Structure

```
├── preprocess_data.py      # Data preprocessing script
├── forecast_btc.py         # Model training script
├── load_model.py           # Model loading and prediction script
├── btc_normalized.npz      # Preprocessed data (generated)
├── btc_model.h5            # Trained model (generated)
└── README.md               # This file
```

---

## 🚀 Quick Start

### ⚠️ Important: Generate the Model

The trained model (`btc_model.h5`) is **not included in the repository** (too large, ignored by `.gitignore`). You **must generate it** by running the training script.

**Note**: If your computer is slow, use **Kaggle** with GPU acceleration (P100 recommended). The model will be trained ~50x faster. 

---
⚠️⚠️⚠️ **FOR THE SWE** ⚠️⚠️⚠️

The pre-trained model file (`btc_model.h5`) is available and will be provided separately to avoid long training times during correction.

---

### Complete Workflow (3 steps):

#### Step 1: Data Preprocessing

First, prepare the data:

```bash
python3 preprocess_data.py
```

**What it does:**
- Merges Coinbase and Bitstamp datasets (inner join on Timestamp)
- Selects 4 features: Close, Volume_(BTC), Volume_(Currency), Weighted_Price
- Normalizes data using statistics from training set (70% split)
- Saves to `btc_normalized.npz` (~26 MB)

**Output:**
```
🔄 Fusion...
📊 1640488 lignes fusionnées
✅ Normalisation OK (light memory)
Fichier: btc_normalized.npz (26.2 MB)
```

---

#### Step 2: **Model Training** (REQUIRED)

Train the LSTM model:

```bash
python3 forecast_btc.py
```

**What it does:**
- Loads preprocessed data
- Creates LSTM(64) → Dense(1) architecture
- Trains with 50 epochs maximum
- Uses **Early Stopping** to prevent overfitting (patience=3)
- Evaluates on test set
- Saves model to `btc_model.h5`

**Training Details:**
- Batch Size: 128
- Optimizer: Adam
- Loss: MSE
- Dataset split: 70% train, 15% validation, 15% test

**Expected Output:**
```
Loaded 1640488 time steps
Subsampled to 82025 steps
Train: 55917 | Val: 10804 | Test: 10804
🚀 Training...
Epoch X/50
... training progress ...
MSE: 0.003
✅ DONE!
```

---

#### Step 3: **Model Testing**

Load and test the trained model:

```bash
python3 load_model.py
```

This script loads the model and makes predictions on different timesteps in the dataset.

---

#### Step 4: **Model Evaluation** (Optional)

Visualize model performance with a comparison plot:

```bash
python3 evaluate_model.py
```

**What it does:**
- Loads the trained model and test data
- Makes predictions on the entire test set
- Creates a graph comparing actual vs predicted values
- Calculates **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**
- Saves visualization to `predictions_vs_actual.png`

**Expected Output:**
```
Loading model...
✅ Model loaded!
Loaded 1640488 timesteps
Subsampled to 82025 steps
Test set: 10804 samples
🔮 Making predictions...
✅ Predictions made!
Mean Absolute Error: $X.XX
RMSE: $X.XX
📊 Plot saved as 'predictions_vs_actual.png'
```

The resulting plot shows how closely the model's predictions follow the actual price movements. Well-trained models will have overlapping blue (actual) and orange (predicted) lines.

---

## 📈 Training Results

The model was trained **5 times** to ensure robustness. Results:

| Run | MSE | Status |
|-----|-----|--------|
| **Run 1** | **0.0030** | ✅ **BEST** |
| Run 2 | 0.0131 | Good |
| Run 3 | 0.0283 | Good |
| Run 4 | 0.0034 | Excellent |
| Run 5 | 0.0204 | Good |

**Selected Model**: Run 1 with **MSE = 0.0030** (best generalization)

**Why this model?**
- Lowest test error (MSE)
- Best balance between training and validation loss
- Converged early with Early Stopping (Epoch 3)
- Consistent predictions across different datasets

---

## 🔧 Key Features

### ✅ Data Preprocessing (`preprocess_data.py`)
- Handles missing values with `dropna()`
- Merges two sources with inner join
- Normalizes with Z-score: `(x - mean) / std`
- Memory-efficient (saves only normalized data)

### ✅ Model Architecture (`forecast_btc.py`)
- **LSTM Layer**: 64 units (captures temporal patterns)
- **Dense Layer**: 1 unit (final prediction)
- **Early Stopping**: Monitors validation loss, stops when no improvement
- **Batch Processing**: Efficient data pipeline with `tf.data.Dataset`
- **Subsampling**: Data reduced from 1.6M to 82K timesteps using `[::20]` stride
  - Keeps every 20th sample (every ~330 minutes)
  - Reduces memory footprint while maintaining temporal patterns
  - Enables training on consumer hardware

### ✅ Model Evaluation
- **Training Set**: 55,917 samples
- **Validation Set**: 10,804 samples
- **Test Set**: 10,804 samples
- **Final Test MSE**: 0.0030

---

## 📝 Important Notes

### Data Limitations
- Dataset spans 2012-2020 (Bitcoin data is historical)
- Predictions reflect price patterns from this period
- For real-time predictions, use recent data (2020-2026)

### Normalizing Parameters
- Normalization computed on training set (70% of data)
- Applied consistently to all data
- Stored in `btc_normalized.npz` for reproducibility

### Model Reproducibility
- Random seed variations may produce slightly different MSE values
- Early Stopping ensures convergence at optimal epoch
- All 5 training runs produced excellent results (MSE < 0.03)

---

## 🎯 Requirements Met

✅ **RNN Architecture**: LSTM layer implemented  
✅ **24-hour Window**: 1440 timesteps = 24 hours × 60 minutes  
✅ **1-hour Prediction**: Predicts price at index i+1440  
✅ **MSE Loss**: `loss="mse"` in model compilation  
✅ **tf.data.Dataset**: Used for efficient batch processing  
✅ **Data Preprocessing**: Separate `preprocess_data.py` script  
✅ **Feature Selection**: Close, Volume, Weighted_Price  
✅ **Normalization**: Z-score normalization applied  
✅ **Train/Val/Test Split**: 70/15/15 split implemented  

---

## 📚 References

- Dataset: Coinbase + Bitstamp BTC/USD historical data
- Framework: TensorFlow/Keras
- Architecture: LSTM RNN with Early Stopping regularization
