from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from main import TransformerModAdd


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)

    hp = ckpt["model_hparams"]
    model = TransformerModAdd(
        P=hp["P"],
        d_model=hp["d_model"],
        n_head=hp["n_head"],
        d_fc=hp["d_fc"],
        seq_len=hp["seq_len"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def plot_metrics(result_path: str = "results"):
    fig, ax = plt.subplots(figsize=(8,5), nrows=2, ncols=1)

    paths = sorted(Path(result_path).glob("metrics/seed*.npz"))
    if not paths:
        raise FileNotFoundError(f"No metric files found in {result_path}")

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for p in paths:
        data = np.load(p)
        train_loss, train_acc, test_loss, test_acc = (
            data["train_loss"],
            data["train_acc"],
            data["test_loss"],
            data["test_acc"],
        )

        ax[0].plot(train_loss, color="blue", alpha=0.3)
        ax[0].plot(test_loss, color="red", alpha=0.3)

        ax[1].plot(train_acc, color="blue", alpha=0.3)
        ax[1].plot(test_acc, color="red", alpha=0.3)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    avg_train_loss = np.mean(np.stack(train_losses, axis=0), axis=0)
    avg_test_loss = np.mean(np.stack(test_losses, axis=0), axis=0)
    avg_train_acc = np.mean(np.stack(train_accs, axis=0), axis=0)
    avg_test_acc = np.mean(np.stack(test_accs, axis=0), axis=0)

    ax[0].plot(avg_train_loss, color="blue", label="Average Train Loss")
    ax[0].plot(avg_test_loss, color="red", label="Average Test Loss")

    ax[1].plot(avg_train_acc, color="blue", label="Average Train Accuracy")
    ax[1].plot(avg_test_acc, color="red", label="Average Test Accuracy")
    
    ax[0].set_yscale("log")
    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower left")

    plt.savefig("plots/fig1_metrics.pdf")


def _fft_norms(w, p, norm="ortho"):
    assert w.shape[0] == p

    wf = torch.fft.rfft(w.to(torch.float32), dim=0, norm=norm)
    cos_norm = torch.linalg.vector_norm(wf.real, ord=2, dim=tuple(range(1, wf.ndim)))
    sin_norm = torch.linalg.vector_norm(wf.imag, ord=2, dim=tuple(range(1, wf.ndim)))

    return cos_norm, sin_norm


def fourier_components(result_path: str = "results"):

    path = os.path.join(result_path, "checkpoints", "modadd_seed5.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metric files found in {result_path}")

    model, ckpt = load_checkpoint(path)
    we = model.we().detach()
    wl = model.wl().detach()

    # drop the "="
    P = we.shape[0] - 1
    assert P == 113  # check if implementation is correct
    assert wl.shape[0] == P
    we = we[:P]  # shape (P, d_model)

    we_cos_norm, we_sin_norm = _fft_norms(we, P)
    wl_cos_norm, wl_sin_norm = _fft_norms(wl, P)

    def _local_peaks(x):
        return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    
    def _top_peaks(w, k):
        peaks = _local_peaks(w)
        scores = w[peaks]
        top_idx = np.argsort(scores)[-k:][::-1]
        top_peaks = peaks[top_idx]
        return top_peaks.tolist()
    
    top_peaks_we_cos = _top_peaks(we_cos_norm.numpy(), 7)
    top_peaks_we_sin = _top_peaks(we_sin_norm.numpy(), 7)
    top_peaks_wl_cos = _top_peaks(wl_cos_norm.numpy(), 7)
    top_peaks_wl_sin = _top_peaks(wl_sin_norm.numpy(), 7)

    print(rf"Top 7 peaks, $W_E$ cos: {top_peaks_we_cos}")
    print(rf"Top 7 peaks, $W_E$ sin: {top_peaks_we_sin}")
    print(rf"Top 7 peaks, $W_L$ cos: {top_peaks_wl_cos}")
    print(rf"Top 7 peaks, $W_L$ sin: {top_peaks_wl_sin}")

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    k = np.arange(len(we_cos_norm))

    ax[0].plot(k, we_cos_norm.numpy(), label="cos", color="red")
    ax[0].plot(k, we_sin_norm.numpy(), label="sin", color="blue")
    ax[0].set_title("Embedding Matrix")
    ax[0].set_xlabel("Frequency k")
    ax[0].set_ylabel("Norm of Fourier Components")
    ax[0].legend()

    ax[1].plot(k, wl_cos_norm.numpy(), label="cos", color="red")
    ax[1].plot(k, wl_sin_norm.numpy(), label="sin", color="blue")
    ax[1].set_title("Neuron-Logit Map")
    ax[1].set_xlabel("Frequency k")
    ax[1].set_ylabel("Norm of Fourier Components")
    ax[1].legend()

    plt.savefig("plots/fig2_fourier.pdf")

def analyze_unembd(result_path: str = "results"):

    path = os.path.join(result_path, "checkpoints", "modadd_seed5.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metric files found in {result_path}")

    model, ckpt = load_checkpoint(path)

    wl = model.wl().detach()
    s = torch.linalg.svdvals(wl)
    s = s.float().cpu()

    energy = (s ** 2)
    cum_energy = torch.cumsum(energy, dim=0) / torch.sum(energy)

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    ax[0].plot(range(1, len(s) + 1)[:25], s.numpy()[:25], marker="o")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("component index")
    ax[0].set_ylabel("singular value")
    ax[0].set_title(r"Scree plot of $W_L$")

    ax[1].plot(range(1, len(cum_energy) + 1)[:25], cum_energy.numpy()[:25], marker="o")
    ax[1].set_xlabel("rank")
    ax[1].set_ylabel("cumulative explained energy")
    ax[1].set_title(r"explained energy vs. rank of $W_L$")
    ax[1].set_ylim(0, 1.01)

    plt.savefig("plots/fig3_umembedding.pdf")

def plot_metrics_progress_measure(result_path: str="results"):

    path = os.path.join(result_path, "progress", "progress_seed5.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metric files found in {result_path}")
    
    data = np.load(path)
    n_points = len(data["restricted_loss"])
    epoch_axis = np.linspace(0, 40000, n_points)
    
    fig, ax = plt.subplots(figsize=(10,5), nrows=2, ncols=2, constrained_layout=True)

    # excluded loss
    _ax = ax[0,0]
    _ax.plot(epoch_axis, data["train_loss"], label="train loss")
    _ax.plot(epoch_axis, data["test_loss"], label="test loss")
    _ax.plot(epoch_axis, data["excluded_loss"], label="excluded loss")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.set_title("Excluded Loss")

    # restricted loss
    _ax = ax[0,1]
    _ax.plot(epoch_axis, data["train_loss"], label="train loss")
    _ax.plot(epoch_axis, data["test_loss"], label="test loss")
    _ax.plot(epoch_axis, data["restricted_loss"], label="restricted loss")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.set_title("Restricted Loss")

    # gini coefficients
    _ax = ax[1,0]
    _ax.plot(epoch_axis, data["gini_we"], label=r"$W_E$")
    _ax.plot(epoch_axis, data["gini_wl"], label=r"$W_L$")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Gini Coefficient")
    _ax.set_title("Gini Coefficients")

    # total sum
    _ax = ax[1,1]
    _ax.plot(epoch_axis, data["total_squared_weight"])
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Sum of Squared Weights")
    _ax.set_title("Total Sum of Squared Weights")

    plt.savefig("plots/fig4_progressmeasure.pdf")

def plot_metrics_progress_measure_wd0(result_path: str="results_wd0"):

    path = os.path.join(result_path, "progress", "progress_seed5.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metric files found in {result_path}")
    
    data = np.load(path)
    n_points = len(data["restricted_loss"])
    epoch_axis = np.linspace(0, 40000, n_points)
    
    fig, ax = plt.subplots(figsize=(10,5), nrows=2, ncols=2, constrained_layout=True)

    # excluded loss
    _ax = ax[0,0]
    _ax.plot(epoch_axis, data["train_loss"], label="train loss")
    _ax.plot(epoch_axis, data["test_loss"], label="test loss")
    _ax.plot(epoch_axis, data["excluded_loss"], label="excluded loss")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.set_title("Excluded Loss, wd=0")

    # restricted loss
    _ax = ax[0,1]
    _ax.plot(epoch_axis, data["train_loss"], label="train loss")
    _ax.plot(epoch_axis, data["test_loss"], label="test loss")
    _ax.plot(epoch_axis, data["restricted_loss"], label="restricted loss")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.set_title("Restricted Loss, wd=0")

    # gini coefficients
    _ax = ax[1,0]
    _ax.plot(epoch_axis, data["gini_we"], label=r"$W_E$")
    _ax.plot(epoch_axis, data["gini_wl"], label=r"$W_L$")
    _ax.legend(loc="best")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Gini Coefficient")
    _ax.set_title("Gini Coefficients, wd=0")

    # total sum
    _ax = ax[1,1]
    _ax.plot(epoch_axis, data["total_squared_weight"])
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Sum of Squared Weights")
    _ax.set_title("Total Sum of Squared Weights, wd=0")

    plt.savefig("plots/fig5_progressmeasure_wd0.pdf")
