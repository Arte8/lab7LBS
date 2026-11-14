"""Reproduce Lakeshore Bike Share training, baselines, experiments, and figures.

Run from the project directory: python train.py
Requires the instructor-supplied bikes_data.py beside this file.
The held-out test set is evaluated once after validation-based selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from bikes_data import make_rentals
from model import BikeRentalNN
from prepare import (
    SEED,
    make_loaders, prepare_data,
)

EXPERIMENTS = [
    {"name": "Main", "hidden1": 64, "hidden2": 32, "lr": 0.001},
    {"name": "Lower Learning Rate", "hidden1": 64, "hidden2": 32, "lr": 0.0005},
    {"name": "Larger Hidden Layer", "hidden1": 128, "hidden2": 32, "lr": 0.001},
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU execution avoids device-specific differences; single-thread execution
    # makes results more stable across repeated runs on the same installation.
    torch.set_num_threads(1)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = y_batch.size(0)
            total_loss += loss.item() * n
            total_absolute_error += torch.abs(predictions - y_batch).sum().item()
            total_samples += n
    if total_samples == 0:
        raise ValueError("Cannot run an epoch on an empty DataLoader")
    return total_loss / total_samples, total_absolute_error / total_samples


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100):
    history = []
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
        )
    model.load_state_dict(best_weights)
    print(f"Best epoch: {best_epoch} | Lowest validation loss: {best_loss:.4f}")
    return pd.DataFrame(history)


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_hourly_scatter(df, reports):
    plt.figure(figsize=(10, 5))
    plt.scatter(df["hour"], df["rentals"], alpha=0.3, color="blue")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Rentals")
    plt.title("Lakeshore Bike Share: Rentals by Hour")
    plt.grid(True, alpha=0.3)
    save_figure(reports / "rentals_by_hour.png")


def plot_learning_curve(history, reports):
    best_row = history.loc[history["val_loss"].idxmin()]
    best_epoch = int(best_row["epoch"])
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss", color="blue")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", color="orange")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.scatter(best_epoch, best_row["val_loss"], color="red", zorder=5)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Lakeshore Bike Share - Neural Network Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(reports / "learning_curves.png")

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["val_mae"], label="Validation MAE", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (Rentals)")
    plt.title("Validation MAE Across Training Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(reports / "validation_mae.png")


def plot_baselines(comparison, reports):
    plt.figure(figsize=(9, 5))
    plt.bar(comparison["Model"], comparison["Validation MAE"])
    plt.xlabel("Model")
    plt.ylabel("Mean Absolute Error (Rentals)")
    plt.title("Bike Rental Prediction - Validation MAE")
    plt.grid(axis="y", alpha=0.3)
    save_figure(reports / "baseline_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run (default: 100)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    set_seed()
    reports = Path(__file__).resolve().parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    df = make_rentals()
    print("Dataset shape:", df.shape)
    print(df["rentals"].describe().round(2))
    plot_hourly_scatter(df, reports)

    prepared = prepare_data(df, seed=SEED)
    y_train, y_val, y_test = (prepared[k] for k in ("y_train", "y_val", "y_test"))
    X_train_prepared = prepared["X_train"]
    X_val_prepared = prepared["X_val"]
    print(
        f"Rows: train={len(y_train)}, validation={len(y_val)}, "
        f"held-out test={len(y_test)}"
    )
    loss_fn = nn.MSELoss()

    # Both baselines use exactly the same training and validation rows.
    mean_predictions = np.full(len(y_val), y_train.mean())
    linear_model = LinearRegression()
    linear_model.fit(X_train_prepared, y_train)
    linear_predictions = linear_model.predict(X_val_prepared)

    best_model = None
    best_config = None
    best_validation_loss = float("inf")
    main_baselines = None
    experiment_rows = []
    histories = {}
    for config in EXPERIMENTS:
        print(f"\n=== {config['name']} ===")
        set_seed()
        # An independent generator ensures identical training batch order for
        # runs with the same number of samples and batch size.
        train_loader, val_loader, _ = make_loaders(prepared, seed=SEED)
        model = BikeRentalNN(
            X_train_prepared.shape[1], config["hidden1"], config["hidden2"]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        history = train_model(model, train_loader, val_loader, loss_fn, optimizer, args.epochs)
        histories[config["name"]] = history
        if float(history["val_loss"].min()) < best_validation_loss:
            best_validation_loss = float(history["val_loss"].min())
            best_config = config
            best_model = copy.deepcopy(model)
        best_row = history.loc[history["val_loss"].idxmin()]
        experiment_rows.append({
            "Run": config["name"],
            "Hidden Layers": f"{config['hidden1']}, {config['hidden2']}",
            "Learning Rate": config["lr"],
            "Best Epoch": int(best_row["epoch"]),
            "Validation Loss": float(best_row["val_loss"]),
            "Validation MAE": float(best_row["val_mae"]),
        })
        slug = config["name"].lower().replace(" ", "_")
        history.to_csv(reports / f"history_{slug}.csv", index=False)
        if config["name"] == "Main":
            history.to_csv(reports / "history.csv", index=False)
            plot_learning_curve(history, reports)
            nn_mse, nn_mae = run_epoch(model, val_loader, loss_fn)
            comparison = pd.DataFrame({
                "Model": ["Training Mean", "Linear Regression", "Neural Network"],
                "Validation MAE": [
                    mean_absolute_error(y_val, mean_predictions),
                    mean_absolute_error(y_val, linear_predictions),
                    nn_mae,
                ],
                "Validation MSE": [
                    mean_squared_error(y_val, mean_predictions),
                    mean_squared_error(y_val, linear_predictions),
                    nn_mse,
                ],
            }).sort_values("Validation MAE").reset_index(drop=True)
            main_baselines = comparison
            comparison.to_csv(reports / "baseline_comparison.csv", index=False)
            plot_baselines(comparison, reports)
            print("\nValidation baseline comparison:\n", comparison.round(4).to_string(index=False))

    experiment_comparison = pd.DataFrame(experiment_rows)
    experiment_comparison.to_csv(reports / "experiments.csv", index=False)
    experiment_comparison.to_csv(reports / "experiment_comparison.csv", index=False)
    print("\nExperiment comparison:\n", experiment_comparison.round(4).to_string(index=False))
    best = experiment_comparison.loc[experiment_comparison["Validation Loss"].idxmin()]
    print(f"\nBest validation configuration: {best['Run']} (epoch {best['Best Epoch']})")
    print(f"Reports saved in: {reports}")
    # Final evaluation: only now, after selecting the best model on validation.
    assert best_model is not None and best_config is not None
    _, _, test_loader = make_loaders(prepared, seed=SEED)
    test_mse, test_mae = run_epoch(best_model, test_loader, loss_fn)
    test_mean_predictions = np.full(len(y_test), y_train.mean())
    test_linear_predictions = linear_model.predict(prepared["X_test"])
    metrics = {
        "seed": SEED,
        "epochs_per_run": args.epochs,
        "rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "selected_model": best_config["name"],
        "selected_hidden_layers": [best_config["hidden1"], best_config["hidden2"]],
        "selected_learning_rate": best_config["lr"],
        "selected_best_epoch": int(best["Best Epoch"]),
        "selected_validation_mse": float(best["Validation Loss"]),
        "selected_validation_mae": float(best["Validation MAE"]),
        "validation_baselines": {
            row["Model"]: {
                "mse": float(row["Validation MSE"]),
                "mae": float(row["Validation MAE"]),
            }
            for _, row in main_baselines.iterrows()
        },
        "test": {
            "neural_network": {"mse": test_mse, "mae": test_mae},
            "training_mean": {
                "mse": float(mean_squared_error(y_test, test_mean_predictions)),
                "mae": float(mean_absolute_error(y_test, test_mean_predictions)),
            },
            "linear_regression": {
                "mse": float(mean_squared_error(y_test, test_linear_predictions)),
                "mae": float(mean_absolute_error(y_test, test_linear_predictions)),
            },
        },
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Final held-out test MAE ({best_config['name']}): {test_mae:.4f} rentals")
    print("Saved reports/metrics.json (test evaluated only after model selection).")


if __name__ == "__main__":
    main()
