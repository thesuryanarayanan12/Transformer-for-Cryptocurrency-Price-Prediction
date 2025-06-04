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

## Model Overview

The model is built using a Transformer Encoder architecture. It leverages a multi-head self-attention mechanism to learn patterns in time series data and capture the importance of historical data points for future predictions.

Refer to the `crypto_predictor.py` file for the model implementation. It includes:

* `PositionalEncoding` for temporal awareness
* `CryptoTransformer` class defining the main Transformer Encoder structure
