from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(result_path: str="results"):
    fig, ax = plt.subplots(nrows=2, ncols=1)
        
    paths = sorted(Path(result_path).glob("metric_seed*.npz"))
    for p in paths:
        data = np.load(p)
        train_loss, train_acc, test_loss, test_acc = data["train_loss"], data["train_acc"], data["test_loss"], data["test_acc"]

        ax[0].plot(train_loss)
        ax[0].plot(test_loss)

        ax[1].plot(train_acc)
        ax[1].plot(test_acc)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/metrics.png", dpi=300)

def analyze_weights(result_path: str="results"):
    pass

if __name__ == "__main__":
    plot_metrics()