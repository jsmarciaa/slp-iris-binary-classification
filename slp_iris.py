from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# konstanta yang digunakan sesuai dengan Google Sheet
LEARNING_RATE = 0.1
EPOCHS = 5
INIT_WEIGHT = 0.5
N_TRAIN_PER_CLASS = 40
N_VAL_PER_CLASS = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "iris_binary.csv"
OUTPUT_PATH = BASE_DIR

# load dan split data menjadi training dan validation set
def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)

    df["target"] = df["species"].map({"Iris-setosa": 0, "Iris-versicolor": 1})    
    if df["target"].isna().any():
            raise ValueError("Unknown species found in the dataset.")

    features = ["X1","X2","X3","X4"]
    setosa = df[df["target"] == 0].reset_index(drop=True)
    versicolor = df[df["target"] == 1].reset_index(drop=True)

    train = pd.concat(
        [setosa.iloc[:N_TRAIN_PER_CLASS],
        versicolor.iloc[:N_TRAIN_PER_CLASS]],
        ignore_index=True,
    )

    val = pd.concat(
        [setosa.iloc[N_TRAIN_PER_CLASS:], 
        versicolor.iloc[N_TRAIN_PER_CLASS:]],
        ignore_index=True,
    )

    x_train = train[features].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    x_val = val[features].to_numpy(dtype=float)
    y_val = val["target"].to_numpy(dtype=int)

    return x_train, y_train, x_val, y_val

# nilai sigmoid
def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def forward(x: np.ndarray, bias: float, theta: np.ndarray) -> float:
    z = bias + float(np.dot(x, theta))
    g = sigmoid(z)
    pred = 1 if g >= 0.5 else 0
    return z, g, pred

# data validation
def evaluate(x: np.ndarray, y: np.ndarray, bias: float, theta: np.ndarray):
    correct, sse = 0, 0.0
    for xi, yi, in zip(x, y):
        _, g, pred = forward(xi, bias, theta)
        correct += int(pred == yi)
        sse += (g - yi) ** 2
    return correct / len(y), sse / len(y)

# data training: loop luar 5 epoch, loop dalam 80 data training
def train_slp(X_train, y_train, X_val, y_val, learning_rate=LEARNING_RATE, epochs=EPOCHS, verbose=True):
    bias = INIT_WEIGHT
    theta = np.full(X_train.shape[1], INIT_WEIGHT, dtype=float)
 
    history = {
        "epoch": [], "train_accuracy": [], "train_loss": [],
        "val_accuracy": [], "val_loss": [],
    }
 
    for epoch in range(1, epochs + 1):
        correct, sse = 0, 0.0
 
        for xi, yi in zip(X_train, y_train):
            _, g, pred = forward(xi, bias, theta)
 
            correct += int(pred == yi)
            sse += (g - yi) ** 2
 
            d = 2.0 * (g - yi) * (1.0 - g) * g
            bias -= learning_rate * d
            theta -= learning_rate * d * xi
 
        train_acc = correct / len(y_train)
        train_loss = sse / len(y_train)
 
        val_acc, val_loss = evaluate(X_val, y_val, bias, theta)
 
        history["epoch"].append(epoch)
        history["train_accuracy"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(val_loss)
 
        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_acc: {train_acc:.4f}  train_loss: {train_loss:.6f} | "
                f"val_acc: {val_acc:.4f}  val_loss: {val_loss:.6f}"
            )
 
    if verbose:
        print("\nFinal weights")
        print(f"  bias   = {bias: .6f}")
        for i, t in enumerate(theta, start=1):
            print(f"  theta{i} = {t: .6f}")
 
    return history, bias, theta

# membuat grafik akurasi dan loss per epoch
def plot_metric(history, key_train, key_val, title, ylabel, filename, ymin=None):
    epochs = history["epoch"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history[key_train], marker="o", label="Training")
    plt.plot(epochs, history[key_val], marker="s", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    if ymin is not None:
        plt.ylim(0, 1.0)
    plt.tight_layout()
    out = OUTPUT_PATH / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")
 
def save_outputs(history):
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_PATH / "metrics.csv", index=False)
    print(f"Saved {OUTPUT_PATH / 'metrics.csv'}")
 
    plot_metric(history, "train_accuracy", "val_accuracy",
                "ACCURACY CHART", "Accuracy", "accuracy_chart.png", ymin=0)
    plot_metric(history, "train_loss", "val_loss",
                "LOSS CHART", "Loss", "loss_chart.png")
 
def main():
    x_train, y_train, x_val, y_val = load_data()
    print(f"Training samples  : {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Learning rate     : {LEARNING_RATE}")
    print(f"Epochs            : {EPOCHS}\n")
 
    history, bias, theta = train_slp(x_train, y_train, x_val, y_val)
    save_outputs(history)
 
    print("\nPer-epoch summary:")
    print(pd.DataFrame(history).to_string(index=False))
 
 
if __name__ == "__main__":
    main()