import os
import json
import glob
import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def load_logs_for_mode(
    log_dir: str,
    mode: str,  # "adv" or "norm"
) -> Dict[str, List[dict]]:
    """
    Load all JSON log files for a given mode ('adv' or 'norm') from log_dir.

    Expects filenames of the form:
        training_<mode>_seed_<seed>.json

    Returns
    -------
    logs_by_seed : dict
        Mapping seed (string) -> list of epoch dicts.
    """
    pattern = os.path.join(log_dir, f"training_{mode}_seed_*.json")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files matching '{pattern}'")

    logs_by_seed: Dict[str, List[dict]] = {}

    for log_path in log_files:
        with open(log_path, "r") as f:
            data = json.load(f)

        base = os.path.basename(log_path)               # training_adv_seed_19302.json
        name, _ = os.path.splitext(base)                # training_adv_seed_19302
        parts = name.split("_")                         # ['training', 'adv', 'seed', '19302']
        if len(parts) == 4:
            seed = parts[3]
        else:
            # fallback if the pattern is somehow different
            seed = name

        logs_by_seed[seed] = data

    return logs_by_seed


def get_epochs(logs_by_seed: Dict[str, List[dict]]) -> np.ndarray:
    """
    Extract the list of epochs (assumes same epochs across seeds).
    """
    first_seed = next(iter(logs_by_seed))
    logs = sorted(logs_by_seed[first_seed], key=lambda d: d["epoch"])
    return np.array([d["epoch"] for d in logs], dtype=int)


def stack_metric(logs_by_seed: Dict[str, List[dict]], key: str) -> np.ndarray:
    """
    Stack one metric across seeds.

    Returns
    -------
    values : np.ndarray
        Array of shape (n_seeds, n_epochs), with NaN where metric is None.
    """
    all_arrays = []
    for logs in logs_by_seed.values():
        logs_sorted = sorted(logs, key=lambda d: d["epoch"])
        vals = [entry.get(key, None) for entry in logs_sorted]
        vals = [np.nan if v is None else v for v in vals]
        all_arrays.append(np.array(vals, dtype=float))

    return np.vstack(all_arrays)  # (n_seeds, n_epochs)


def _mean_std_over_epochs(
    epochs: np.ndarray,
    values: np.ndarray,
):
    """
    Helper: compute (epochs_used, mean, std) ignoring all-NaN epochs.
    """
    mask = ~np.all(np.isnan(values), axis=0)
    epochs_used = epochs[mask]
    values_used = values[:, mask]

    mean = np.nanmean(values_used, axis=0)
    std = np.nanstd(values_used, axis=0)
    return epochs_used, mean, std


def plot_loss_adv_vs_norm(
    epochs: np.ndarray,
    loss_adv: np.ndarray,
    loss_norm: np.ndarray,
    out_path: str,
) -> None:
    """
    Plot training loss vs epoch with mean ± std for:
      - adversarial training (solid line),
      - standard training (dashed line).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Adversarial training
    ep_adv, mean_adv, std_adv = _mean_std_over_epochs(epochs, loss_adv)
    line_adv, = ax.plot(ep_adv, mean_adv, label="adv train", linewidth=2)
    color_adv = line_adv.get_color()
    ax.fill_between(
        ep_adv, mean_adv - std_adv, mean_adv + std_adv,
        alpha=0.2, color=color_adv
    )

    # Standard (non-adversarial) training
    ep_norm, mean_norm, std_norm = _mean_std_over_epochs(epochs, loss_norm)
    line_norm, = ax.plot(
        ep_norm, mean_norm,
        linestyle="--", linewidth=2, label="standard train"
    )
    color_norm = line_norm.get_color()
    ax.fill_between(
        ep_norm, mean_norm - std_norm, mean_norm + std_norm,
        alpha=0.2, color=color_norm
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (NLL)")
    ax.set_title("Training loss vs epoch (adv vs standard, mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_acc_adv_vs_norm(
    epochs: np.ndarray,
    acc_nat_adv: np.ndarray,
    acc_pgd_linf_adv: np.ndarray,
    acc_pgd_l2_adv: np.ndarray,
    acc_nat_norm: np.ndarray,
    acc_pgd_linf_norm: np.ndarray,
    acc_pgd_l2_norm: np.ndarray,
    out_path: str,
) -> None:
    """
    Combined accuracy plot comparing adversarial vs non-adversarial training.

    - For each metric (natural, PGD-ℓ∞, PGD-ℓ₂):
        * solid line + band  -> adv training
        * dashed line + band -> standard training
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    def _plot_metric(values_adv, values_norm, label_base, color_idx: int):
        # choose color from default cycle by plotting an invisible point
        tmp_line, = ax.plot([], [], color=f"C{color_idx}")
        color = tmp_line.get_color()
        tmp_line.remove()

        # adv: solid
        ep_adv, mean_adv, std_adv = _mean_std_over_epochs(epochs, values_adv)
        line_adv, = ax.plot(
            ep_adv, mean_adv,
            label=f"{label_base} (adv)", color=color
        )
        ax.fill_between(
            ep_adv, mean_adv - std_adv, mean_adv + std_adv,
            alpha=0.2, color=color
        )

        # norm: dashed
        ep_norm, mean_norm, std_norm = _mean_std_over_epochs(epochs, values_norm)
        ax.plot(
            ep_norm, mean_norm,
            linestyle="--",
            color=color,
            label=f"{label_base} (std)"
        )
        ax.fill_between(
            ep_norm, mean_norm - std_norm, mean_norm + std_norm,
            alpha=0.1, color=color
        )

    _plot_metric(acc_nat_adv,      acc_nat_norm,      "natural",    0)
    _plot_metric(acc_pgd_linf_adv, acc_pgd_linf_norm, "PGD-$\\ell_\\infty$", 1)
    _plot_metric(acc_pgd_l2_adv,   acc_pgd_l2_norm,   "PGD-$\\ell_2$",      2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Natural and adversarial accuracies vs epoch\n"
        "(adv vs standard training, mean ± std across seeds)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare adv vs standard training from JSON logs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory containing training_adv_seed_*.json and training_norm_seed_*.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figs_compare",
        help="Directory to save output figures.",
    )
    args = parser.parse_args()

    # Load logs for both regimes
    logs_adv = load_logs_for_mode(args.log_dir, mode="adv")
    logs_norm = load_logs_for_mode(args.log_dir, mode="norm")

    # We assume same epoch grid for both (same training schedule)
    epochs = get_epochs(logs_adv)

    # Stack metrics
    train_loss_adv = stack_metric(logs_adv, "train_loss")
    acc_nat_adv = stack_metric(logs_adv, "acc_nat")
    acc_pgd_linf_adv = stack_metric(logs_adv, "acc_pgd_linf")
    acc_pgd_l2_adv = stack_metric(logs_adv, "acc_pgd_l2")

    train_loss_norm = stack_metric(logs_norm, "train_loss")
    acc_nat_norm = stack_metric(logs_norm, "acc_nat")
    acc_pgd_linf_norm = stack_metric(logs_norm, "acc_pgd_linf")
    acc_pgd_l2_norm = stack_metric(logs_norm, "acc_pgd_l2")

    # Loss comparison
    out_loss = os.path.join(args.out_dir, "loss_adv_vs_norm.png")
    plot_loss_adv_vs_norm(
        epochs=epochs,
        loss_adv=train_loss_adv,
        loss_norm=train_loss_norm,
        out_path=out_loss,
    )

    # Accuracy comparison
    out_acc = os.path.join(args.out_dir, "acc_adv_vs_norm.png")
    plot_acc_adv_vs_norm(
        epochs=epochs,
        acc_nat_adv=acc_nat_adv,
        acc_pgd_linf_adv=acc_pgd_linf_adv,
        acc_pgd_l2_adv=acc_pgd_l2_adv,
        acc_nat_norm=acc_nat_norm,
        acc_pgd_linf_norm=acc_pgd_linf_norm,
        acc_pgd_l2_norm=acc_pgd_l2_norm,
        out_path=out_acc,
    )

    print(f"Figures saved in '{args.out_dir}'.")


if __name__ == "__main__":
    main()
