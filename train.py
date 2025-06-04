import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from crypto_predictor import CryptoTransformer
import os

# Parameters
input_dim = 5
d_model = 64
nhead = 4
num_layers = 3
dim_feedforward = 256
seq_len = 60
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Directories
data_path = 'data/crypto_data.csv'
model_save_path = 'models/best_model.pth'
results_dir = 'results'
os.makedirs('models', exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(data_path)
df = df[['open', 'high', 'low', 'close', 'volume']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][3]  # predicting 'close' price
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

X, y = create_sequences(scaled_data, seq_len)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Dataloaders
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# Initialize model
model = CryptoTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_save_path)

# Evaluation
model.load_state_dict(torch.load(model_save_path))
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        output = model(batch_x)
        predictions.append(output.numpy())
        actuals.append(batch_y.numpy())
predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Save predictions
pred_close = scaler.inverse_transform(np.hstack([np.zeros((actuals.shape[0], 3)), actuals, np.zeros((actuals.shape[0], 1))]))[:, 3]
pred_output = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], 3)), predictions, np.zeros((predictions.shape[0], 1))]))[:, 3]
result_df = pd.DataFrame({'Actual': pred_close, 'Predicted': pred_output})
result_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(result_df['Actual'], label='Actual')
plt.plot(result_df['Predicted'], label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.savefig(os.path.join(results_dir, 'prediction_plot.png'))
plt.close()
