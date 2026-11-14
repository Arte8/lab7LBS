"""Tests for the first-principles neural network architecture."""
import pytest
import torch
from torch import nn

from model import BikeRentalNN


@pytest.mark.parametrize("hidden1,hidden2", [(64, 32), (128, 32), (32, 16)])
def test_two_relu_hidden_layers_and_linear_output(hidden1, hidden2):
    model = BikeRentalNN(8, hidden1, hidden2)
    layers = list(model.network)
    assert [x.out_features for x in layers if isinstance(x, nn.Linear)] == [hidden1, hidden2, 1]
    assert sum(isinstance(x, nn.ReLU) for x in layers) == 2
    assert isinstance(layers[-1], nn.Linear)
    assert model(torch.zeros(5, 8)).shape == (5, 1)
