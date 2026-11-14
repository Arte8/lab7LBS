"""Custom two-hidden-layer regression network (linear output)."""

from torch import nn


class BikeRentalNN(nn.Module):
    def __init__(self, input_size: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.network(x)
