
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

runs = {
    "Main": "history_main.csv",
    "Lower Learning Rate": "history_lower_learning_rate.csv",
    "Larger Hidden Layer": "history_larger_hidden_layer.csv",
}

for name, filename in runs.items():
    history = pd.read_csv(REPORTS_DIR / filename)

    best_row = history.loc[history["val_loss"].idxmin()]
    final_row = history.iloc[-1]

    print(f"\n{name}")
    print("Best epoch:", int(best_row["epoch"]))
    print("Validation MAE at best epoch:", round(best_row["val_mae"], 4))
    print("Final training MSE:", round(final_row["train_loss"], 4))
    print("Final validation MSE:", round(final_row["val_loss"], 4))
