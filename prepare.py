"""Training-only preparation and PyTorch datasets for Lakeshore Bike Share."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 32
NUMERIC_FEATURES = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL_FEATURES = ["station_type"]


class BikeRentalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def split_data(df, seed=SEED):
    """Split rows 60/20/20 before fitting any preprocessing."""
    train, remaining = train_test_split(df, test_size=0.40, random_state=seed)
    validation, test = train_test_split(remaining, test_size=0.50, random_state=seed)
    return train, validation, test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def prepare_data(df, seed=SEED):
    """Fit preprocessing only on training rows; return aligned prepared splits."""
    train, validation, test = split_data(df, seed)
    X_train, y_train = train.drop(columns="rentals"), train["rentals"]
    X_val, y_val = validation.drop(columns="rentals"), validation["rentals"]
    X_test, y_test = test.drop(columns="rentals"), test["rentals"]
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    return {
        "preprocessor": preprocessor,
        "train_rows": train,
        "val_rows": validation,
        "test_rows": test,
        "X_train": X_train_p,
        "X_val": X_val_p,
        "X_test": X_test_p,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def make_loaders(prepared, seed=SEED, batch_size=BATCH_SIZE):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BikeRentalDataset(prepared["X_train"], prepared["y_train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        BikeRentalDataset(prepared["X_val"], prepared["y_val"]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        BikeRentalDataset(prepared["X_test"], prepared["y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
