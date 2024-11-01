# src/model.py

import torch
from torch import nn

class ImdbLSTM(nn.Module):
    def __init__(self, input_size=5000, lstm_hidden_size=130, lstm_layers=3, fc_size=[64, 32, 16], op_size=1):
        super().__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.fc_size = fc_size
        self.op_size = op_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bias=True,
                            batch_first=True,
                            dropout=0.4,
                            bidirectional=False)

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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.layer_stack(lstm_out[:, -1, :])  # Use the last time step's output
