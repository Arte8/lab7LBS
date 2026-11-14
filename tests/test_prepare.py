"""Tests for deterministic splits, leakage prevention and DataLoaders."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from bikes_data import make_rentals
from prepare import BikeRentalDataset, make_loaders, prepare_data, split_data


def test_generator_reproducible():
    pd.testing.assert_frame_equal(make_rentals(25, 2026), make_rentals(25, 2026))


def test_split_sizes_and_disjoint_indices(small_frame):
    train, val, test = split_data(small_frame)

    assert (len(train), len(val), len(test)) == (72, 24, 24)
    assert len(train) + len(val) + len(test) == len(small_frame)

    train_indices = set(train.index)
    val_indices = set(val.index)
    test_indices = set(test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    assert train_indices | val_indices | test_indices == set(small_frame.index)


def test_preprocessor_outputs_eight_finite_features(small_frame):
    prepared = prepare_data(small_frame)
    assert prepared["X_train"].shape == (72, 8)
    assert prepared["X_val"].shape == (24, 8)
    assert prepared["X_test"].shape == (24, 8)
    assert np.isfinite(prepared["X_val"]).all()


def test_preprocessing_is_fitted_only_on_training_rows(small_frame):
    prepared = prepare_data(small_frame)

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]

    scaler = prepared["preprocessor"].named_transformers_["num"]

    # Training numeric features should be centered around zero.
    assert np.allclose(X_train[:, :5].mean(axis=0), 0, atol=1e-6)

    # Validation data should not be independently standardized.
    assert not np.allclose(X_val[:, :5].mean(axis=0), 0, atol=1e-6)

    # The scaler should be fitted on training rows only.
    assert scaler.n_samples_seen_ == 72


def test_dataset_returns_float32():
    dataset = BikeRentalDataset(np.ones((4, 8)), [1, 2, 3, 4])
    x, y = dataset[0]
    assert len(dataset) == 4
    assert x.shape == (8,) and y.shape == (1,)
    assert x.dtype == y.dtype == torch.float32


def test_loader_shuffle_only_for_training(small_frame):
    train, val, test = make_loaders(prepare_data(small_frame))
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)

def test_training_loader_reproducible_shuffle(small_frame):
    prepared = prepare_data(small_frame)

    # Reset the PyTorch RNG before creating each loader.
    torch.manual_seed(42)
    train_a, _, _ = make_loaders(prepared)

    torch.manual_seed(42)
    train_b, _, _ = make_loaders(prepared)

    # Reset before each iteration to compare batch ordering.
    torch.manual_seed(42)
    order_a = torch.cat([y.flatten() for _, y in train_a])

    torch.manual_seed(42)
    order_b = torch.cat([y.flatten() for _, y in train_b])

    assert torch.equal(order_a, order_b)