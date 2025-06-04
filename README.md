# Transformer-for-Cryptocurrency-Price-Prediction

## Prerequisites

* Python 3.8+
* PyTorch
* Pandas
* Scikit-learn
* Matplotlib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/transformer-crypto-prediction.git
cd transformer-crypto-prediction
```

2. Install dependencies:

```bash
pip install torch pandas scikit-learn matplotlib
```

## Usage

1. Prepare your data:
   Place historical cryptocurrency data files in the `data/` directory.
   Ensure each CSV file contains the following columns: `timestamp`, `open`, `high`, `low`, `close`, and `volume`.

2. Train the model:
   Run the training script to preprocess the data, train the Transformer model, and save the best-performing version to the `models/` directory.

```bash
python src/train.py
```

3. View results:
   After training, you’ll find:

   * `predictions.csv` with the model's forecasts
   * `prediction_plot.png` visualizing the predictions
     These will be located in the `results/` folder.

## Sample Results

```
timestamp,actual_price,predicted_price
2025-05-20 10:00:00,67150.50,67085.25
2025-05-20 11:00:00,67210.00,67165.70
2025-05-20 12:00:00,67355.20,67250.90
2025-05-20 13:00:00,67300.75,67380.15
2025-05-20 14:00:00,67450.00,67310.60
2025-05-20 15:00:00,67510.80,67475.45
2025-05-20 16:00:00,67480.25,67530.00
2025-05-20 17:00:00,67600.00,67495.30 ...
```
## Model Overview

The model is built using a Transformer Encoder architecture. It leverages a multi-head self-attention mechanism to learn patterns in time series data and capture the importance of historical data points for future predictions.

Refer to the `crypto_predictor.py` file for the model implementation. It includes:

* `PositionalEncoding` for temporal awareness
* `CryptoTransformer` class defining the main Transformer Encoder structure
