"""Unit tests for Lakeshore Bike Share training (no full training run)."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import train
from bikes_data import make_rentals


@pytest.fixture
def small_data():
    frame = make_rentals(n=120, seed=2026)
    X = frame.drop(columns="rentals")
    y = frame["rentals"]
    return frame, X, y


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(make_rentals(n=30, seed=2026), make_rentals(n=30, seed=2026))


def test_generator_schema_and_nonnegative_counts(small_data):
    frame, _, _ = small_data
    assert set(frame.columns) == set(train.NUMERIC_FEATURES + train.CATEGORICAL_FEATURES + ["rentals"])
    assert (frame["rentals"] >= 0).all()
    assert pd.api.types.is_integer_dtype(frame["rentals"])


def test_split_is_60_20_20_and_disjoint():
    frame = make_rentals(n=120, seed=2026)
    first, remainder = train_test_split(frame, test_size=0.4, random_state=train.SEED)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=train.SEED)
    assert (len(first), len(validation), len(test)) == (72, 24, 24)
    assert not (set(first.index) & set(validation.index))
    assert not (set(first.index) & set(test.index))
    assert not (set(validation.index) & set(test.index))


def test_preprocessor_uses_training_data_and_eight_features(small_data):
    _, X, _ = small_data
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), train.NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore"), train.CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )
    transformed = preprocessor.fit_transform(X.iloc[:90])
    validation = preprocessor.transform(X.iloc[90:])
    assert transformed.shape == (90, 8)
    assert validation.shape == (30, 8)
    assert np.isfinite(validation).all()


@pytest.mark.parametrize("hidden1,hidden2", [(64, 32), (128, 32), (32, 16)])
def test_network_architecture_and_output_shape(hidden1, hidden2):
    model = train.BikeRentalNN(8, hidden1, hidden2)
    layers = list(model.network)
    assert [layer.out_features for layer in layers if isinstance(layer, nn.Linear)] == [hidden1, hidden2, 1]
    assert sum(isinstance(layer, nn.ReLU) for layer in layers) == 2
    assert isinstance(layers[-1], nn.Linear)  # no output activation
    assert model(torch.zeros(4, 8)).shape == (4, 1)


def test_dataset_returns_float32_single_rows():
    dataset = train.BikeRentalDataset(np.ones((5, 8)), np.arange(5))
    features, target = dataset[0]
    assert len(dataset) == 5
    assert features.shape == (8,)
    assert target.shape == (1,)
    assert features.dtype == target.dtype == torch.float32


def test_loader_shuffling_configuration():
    dataset = train.BikeRentalDataset(np.ones((10, 8)), np.arange(10))
    training = DataLoader(dataset, batch_size=4, shuffle=True)
    validation = DataLoader(dataset, batch_size=4, shuffle=False)
    assert isinstance(training.sampler, RandomSampler)
    assert isinstance(validation.sampler, SequentialSampler)


def test_evaluation_loss_and_mae_are_sample_weighted():
    dataset = train.BikeRentalDataset(np.zeros((5, 1)), [1, 2, 3, 4, 5])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    loss, mae = train.run_epoch(model, loader, nn.MSELoss())
    assert loss == pytest.approx(11.0)
    assert mae == pytest.approx(3.0)


def test_evaluation_does_not_change_weights():
    model = train.BikeRentalNN(8)
    loader = DataLoader(train.BikeRentalDataset(np.ones((6, 8)), np.arange(6)), batch_size=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    train.run_epoch(model, loader, nn.MSELoss())
    assert not model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_training_updates_weights():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    loader = DataLoader(train.BikeRentalDataset(np.ones((8, 1)), np.ones(8)), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()
    train.run_epoch(model, loader, nn.MSELoss(), optimizer)
    assert not torch.equal(before, model.weight)


def test_empty_loader_raises_value_error():
    empty = train.BikeRentalDataset(np.empty((0, 1)), np.empty(0))
    loader = DataLoader(empty, batch_size=2)
    with pytest.raises(ValueError, match="empty DataLoader"):
        train.run_epoch(nn.Linear(1, 1), loader, nn.MSELoss())


def test_best_validation_epoch_weights_are_restored(monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Simulate successive train/validation epochs. The last epoch is worse.
    losses = iter([5.0, 2.0, 4.0])
    saved_best = []

    def fake_run_epoch(model_arg, loader, loss_fn, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                model_arg.weight.fill_(len(saved_best) + 1)
            saved_best.append(model_arg.weight.detach().clone())
            return 1.0, 1.0
        return next(losses), 1.0

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)
    history = train.train_model(model, None, None, nn.MSELoss(), optimizer, epochs=3)
    assert len(history) == 3
    assert history["val_loss"].tolist() == [5.0, 2.0, 4.0]
    assert torch.equal(model.weight, saved_best[1])


def test_training_mean_and_linear_baselines_use_validation_rows(small_data):
    _, X, y = small_data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=train.SEED)
    prep = ColumnTransformer(
        [("num", StandardScaler(), train.NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore"), train.CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )
    x_train = prep.fit_transform(X_train)
    x_val = prep.transform(X_val)
    mean_pred = np.full(len(y_val), y_train.mean())
    linear_pred = LinearRegression().fit(x_train, y_train).predict(x_val)
    assert len(mean_pred) == len(linear_pred) == len(y_val)
    assert np.isfinite(mean_absolute_error(y_val, mean_pred))
    assert np.isfinite(mean_absolute_error(y_val, linear_pred))


def test_plot_files_are_created(tmp_path):
    history = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [5.0, 3.0, 2.0],
        "val_loss": [6.0, 2.0, 4.0],
        "val_mae": [2.0, 1.0, 1.5],
    })
    train.plot_learning_curve(history, tmp_path)
    assert (tmp_path / "learning_curve.png").stat().st_size > 0
    assert (tmp_path / "validation_mae.png").stat().st_size > 0
