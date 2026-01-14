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
    fig, ax = plt.subplots(nrows=2, ncols=1)

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

        ax[0].plot(train_loss, color="blue", alpha=0.5)
        ax[0].plot(test_loss, color="red", alpha=0.5)

        ax[1].plot(train_acc, color="blue", alpha=0.5)
        ax[1].plot(test_acc, color="red", alpha=0.5)

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
    ax[1].set_yscale("log")

    fig.legend(pos="best")

    plt.savefig("plots/fig2_metrics.png", dpi=300)


def _fft_norms(w, p, norm="ortho"):
    assert w.shape[0] == p

    wf = torch.fft.rfft(w.to(torch.float32), dim=0, norm=norm)
    cos_norm = torch.linalg.vector_norm(wf.real, ord=2, dim=tuple(range(1, wf.ndim)))
    sin_norm = torch.linalg.vector_norm(wf.imag, ord=2, dim=tuple(range(1, wf.ndim)))

    return cos_norm, sin_norm


def fourier_components(result_path: str = "results"):

    path = os.path.join(result_path, "weights", "seed_0.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metric files found in {result_path}")

    data = torch.load(path)
    we = data["we"]
    wl = data["wl"]

    # drop the "="
    P = we.shape[0] - 1
    assert P == 113  # check if implementation is correct
    assert wl.shape[0] == P
    we = we[:P]  # shape (P, d_model)

    we_cos_norm, we_sin_norm = _fft_norms(we, P)
    wl_cos_norm, wl_sin_norm = _fft_norms(wl, P)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    k = np.arange(len(we_cos_norm))

    ax[0].plot(k, we_cos_norm.numpy(), label="cos", color="red")
    ax[0].plot(k, we_sin_norm.numpy(), label="sin", color="blue")
    ax[0].set_title("Fourier Components of Embedding Matrix")
    ax[0].set_xlabel("Frequency k")
    ax[0].set_ylabel("Norm of Fourier Components")
    ax[0].legend()

    ax[1].plot(k, wl_cos_norm.numpy(), label="cos", color="red")
    ax[1].plot(k, wl_sin_norm.numpy(), label="sin", color="blue")
    ax[1].set_title("Fourier Components of Neuron-Logit Map")
    ax[1].set_xlabel("Frequency k")
    ax[1].set_ylabel("Norm of Fourier Components")
    ax[1].legend()

    plt.savefig("plots/fig3_fourier.png", dpi=300)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_metrics()
