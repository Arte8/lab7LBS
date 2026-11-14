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
    assert not (set(train.index) & set(val.index))
    assert not (set(train.index) & set(test.index))
    assert not (set(val.index) & set(test.index))


def test_preprocessor_outputs_eight_finite_features(small_frame):
    prepared = prepare_data(small_frame)
    assert prepared["X_train"].shape == (72, 8)
    assert prepared["X_val"].shape == (24, 8)
    assert prepared["X_test"].shape == (24, 8)
    assert np.isfinite(prepared["X_val"]).all()


def test_preprocessing_is_fitted_only_on_training_rows(small_frame):
    prepared = prepare_data(small_frame)
    assert np.isclose(prepared["X_train"][:, 0].mean(), 0, atol=1e-6)
    assert prepared["preprocessor"].transformers_[0][1].n_samples_seen_ == 72


def test_dataset_returns_float32():
    dataset = BikeRentalDataset(np.ones((4, 8)), [1, 2, 3, 4])
    x, y = dataset[0]
    assert x.shape == (8,) and y.shape == (1,)
    assert x.dtype == y.dtype == torch.float32


def test_loader_shuffle_only_for_training(small_frame):
    train, val, test = make_loaders(prepare_data(small_frame))
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
