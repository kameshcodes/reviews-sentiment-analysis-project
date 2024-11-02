from typing import Tuple
from torch import nn
import torch


class ImdbLSTM(nn.Module):
    """
    LSTM-based model for sentiment classification of text reviews.
    """

    def __init__(self, input_size: int = 5000, lstm_hidden_size: int = 130,
                 lstm_layers: int = 3, fc_size: Tuple[int, int, int] = (64, 32, 16),
                 op_size: int = 1) -> None:
        """
        Initializes the LSTM and fully connected layers.

        Args:
            input_size: Size of the input features.
            lstm_hidden_size: Number of hidden units in the LSTM.
            lstm_layers: Number of LSTM layers.
            fc_size: Sizes of the fully connected layers.
            op_size: Size of the output layer.
        """
        super().__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.fc_size = fc_size
        self.op_size = op_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            bias=True,
            batch_first=True,
            dropout=0.4,
            bidirectional=False
        )

        # Fully connected layers
        self.layer_stack = nn.Sequential(
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.fc_size[0]),
            nn.BatchNorm1d(self.fc_size[0]),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.fc_size[0], self.fc_size[1]),
            nn.BatchNorm1d(self.fc_size[1]),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.fc_size[1], self.fc_size[2]),
            nn.BatchNorm1d(self.fc_size[2]),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.fc_size[2], self.op_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, op_size).
        """
        lstm_out, _ = self.lstm(x)
        return self.layer_stack(lstm_out[:, -1, :])  # Use the last time step's output


if __name__ == "__main__":
    pass
