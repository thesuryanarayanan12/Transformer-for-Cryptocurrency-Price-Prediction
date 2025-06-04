# crypto_predictor.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This helps the model understand the order of the time-series data.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.pe[:x.size(0), :]

class CryptoTransformer(nn.Module):
    """
    The main Transformer model for cryptocurrency prediction.
    It uses a Transformer Encoder architecture to process sequential data.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        """
        Args:
            input_dim (int): The number of input features (e.g., OHLCV -> 5).
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(CryptoTransformer, self).__init__()
        self.d_model = d_model

        # Linear layer to project input features to the model's dimension
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
       
        # Positional encoding to add sequence information
        self.positional_encoding = PositionalEncoding(d_model)
       
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expects (batch, seq, feature)
        )
       
        # Stacking multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
       
        # Final linear layer to produce the prediction (predicting 1 value, e.g., next close price)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        """
        Forward pass for the model.
       
        Args:
            src (Tensor): The sequence to the encoder. Shape: (batch_size, seq_len, input_dim).
            src_mask (Tensor, optional): The mask for the src sequence.
       
        Returns:
            Tensor: The model's prediction. Shape: (batch_size, 1).
        """
        # 1. Project input features to the model dimension
        src = self.encoder_input_layer(src) * math.sqrt(self.d_model)
       
        # 2. Add positional encodings
        src = self.positional_encoding(src)
       
        # 3. Pass through the Transformer Encoder
        output = self.transformer_encoder(src, src_mask)
       
        # 4. Use the output of the last time step for prediction
        output = output[:, -1, :]
       
        # 5. Pass through the final linear layer
        output = self.output_layer(output)
       
        return output

# Example of how to create the model
if __name__ == '__main__':
    # Model hyperparameters
    input_dim = 5         # Number of features (e.g., Open, High, Low, Close, Volume)
    d_model = 64          # Embedding dimension
    nhead = 4             # Number of attention heads
    num_layers = 3        # Number of Transformer encoder layers
    dim_feedforward = 256 # Dimension of the feedforward network
    seq_len = 60          # Number of time steps in each input sample
    batch_size = 32       # Number of samples in a batch

    # Create a model instance
    model = CryptoTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=dim_feedforward
    )

    # Create a dummy input tensor to test the model
    # Shape: (batch_size, sequence_length, input_features)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Get the model output
    prediction = model(dummy_input)

    print("Model created successfully.")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model prediction shape: {prediction.shape}") # Expected: (batch_size, 1)
