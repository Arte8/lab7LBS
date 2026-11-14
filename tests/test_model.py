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


def test_optimizer_step_updates_every_layer():
    torch.manual_seed(42)

    model = BikeRentalNN(input_size=8, hidden1=64, hidden2=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    X_batch = torch.randn(16, 8)
    y_batch = torch.randn(16, 1)

    # Save original parameters
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Perform one training step
    optimizer.zero_grad()
    loss = loss_fn(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()

    # Verify every parameter was updated
    for name, param in model.named_parameters():
        assert not torch.equal(before[name], param), f"{name} did not update"
