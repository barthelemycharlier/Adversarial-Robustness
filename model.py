"""
CIFAR-10 robust classifier training.

Overview:

We learn a classifier f_θ : R^{3x32x32} → R^{10} that maps an image x to class.
The loss used throughout is Negative Log-Likelihood (cross-entropy with
`log_softmax`), denoted l(f_θ(x), y).

Two training options are supported:

1) Natural (Empirical Risk Minimization) training

   Minimize the expected loss on clean inputs:
       minimize_θ  E_{(x,y)} [ l(f_θ(x), y) ].

   In the code: 
   For each minibatch we forward x, compute l, backprop, and update θ with SGD.

2) Adversarial (robust) training

   Minimize the robust risk against bounded perturbations δ:
       minimize_θ  E_{(x,y)} [  max_{‖δ‖_∞ ≤ ε}  l(f_θ(x + δ), y) ].

   The inner maximization (adversary) is approximated with Projected Gradient
   Descent (PGD) in input space. For the l∞ case:
       x^{t+1} = Π_{B_∞(x, ε)} ( x^t + alpha · sign(∇_x l(f_θ(x^t), y)) ),
   (we do gradient ascent on the loss and project back to the ε-box around x).
   For l2 PGD, the step uses a unit-l2 normalized gradient instead of `sign`.

   In the code: 
   If `adv_train=True`, each minibatch is first turned into x_adv with PGD-l∞
   using (ε, alpha, T) = (train_eps_linf, train_alpha_linf, train_pgd_steps),
   and the update is performed on (x_adv, y), which empirically approximates
   the robust objective above.

Evaluation:

We report:
  - `acc_nat`: accuracy on clean validation images.
  - `acc_pgd_linf`: accuracy under a white-box PGD-l∞ attack with test
    hyper-params (test_eps_linf, test_alpha_linf, test_pgd_steps).
  - `acc_pgd_l2`: accuracy under a white-box PGD-l2 attack with test
    hyper-params (test_eps_l2, test_alpha_l2, test_pgd_steps).
  - An aggregate score `agg = acc_pgd_linf + acc_pgd_l2`
    (used in the platform grading).

References:

Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks".
(https://arxiv.org/abs/1706.06083)
"""

import sys
import os
import argparse
import logging
import time
import json
from dataclasses import asdict
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scripts.attacks import pgd_linf_attack
from scripts.models import build_backbone
from scripts.set_seed import set_seed
from scripts.configs import TrainingConfig, LOGGER, USE_CUDA, DEVICE
from scripts.data import build_dataloaders
from scripts.evaluate import evaluate_model


class Net(nn.Module):
    """Chosen architecture for CIFAR-10 classification."""

    # This file will be loaded to test the model. 
    # Use --model-file to load/store a different model.
    model_file = "models/default_model.pth"

    def __init__(self, arch="resnet18", activation="gelu") -> None:
        super().__init__()
        self.arch = arch
        self.activation = activation
        self.backbone = build_backbone(arch, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save(self, model_file: str) -> None:
        torch.save(self.state_dict(), model_file)

    def load(self, model_file: str) -> None:
        self.load_state_dict(
            torch.load(model_file, map_location=torch.device(DEVICE))
        )

    def load_for_testing(self, project_dir: str = "./") -> None:
        """This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        """  
        self.load(os.path.join(project_dir, Net.model_file))


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    cfg: TrainingConfig,
) -> float:
    """Train the model for a single epoch, return average training loss."""
    net.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        if cfg.adv_train:
            images_adv = pgd_linf_attack(
                model=net,
                x=images,
                y=labels,
                epsilon=cfg.train_eps_linf,
                alpha=cfg.train_alpha_linf,
                num_iter=cfg.train_pgd_steps
            )
            net.train()  # PGD sets eval, reset to train
            outputs = net(images_adv)
        else:
            outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss


def train_model(
    net: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cfg: TrainingConfig,
) -> List[Dict]:
    """
    Main training loop with per-epoch metrics logged for plotting.
    Returns a list of logs (one dict per epoch).
    """
    def _warmup_lr(epoch):
        """Adjust learning rate during warmup period"""
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler_cosine.step()

    def _is_early_stop(
        acc_nat: float,
        acc_pgd_linf: float,
        acc_pgd_l2: float,
        epoch: int,
    ) -> bool:
        """Update early stopping state based on current metrics.

        Metric controlled by cfg.early_stop_metric:
        - "acc_nat"
        - "acc_pgd_linf"
        - "acc_pgd_l2"
        - "agg"  -> acc_pgd_linf + acc_pgd_l2
        - "balanced" -> 2 / (1/acc_nat + 1/(acc_pgd_l2 + acc_pgd_linf))
        """
        nonlocal best_metric, patience_counter, best_state

        # Early stopping disabled, or no fresh metrics this epoch
        if not cfg.early_stop or acc_nat is None:
            return False

        # Select monitored metric
        if cfg.early_stop_metric == "acc_nat":
            current = acc_nat
        elif cfg.early_stop_metric == "acc_pgd_linf":
            if acc_pgd_linf is None:
                return False
            current = acc_pgd_linf
        elif cfg.early_stop_metric == "acc_pgd_l2":
            if acc_pgd_l2 is None:
                return False
            current = acc_pgd_l2
        elif cfg.early_stop_metric == "agg":
            if acc_pgd_linf is None or acc_pgd_l2 is None:
                return False
            current = acc_pgd_linf + acc_pgd_l2
        elif cfg.early_stop_metric == "balanced":
            if acc_pgd_linf is None or acc_pgd_l2 is None:
                return False
            current = 2 / (1/acc_nat + 2/(acc_pgd_l2 + acc_pgd_linf))
        else:
            raise ValueError(
                f"Unsupported early_stop_metric: {cfg.early_stop_metric}"
            )

        # First improvement or better metric
        if current > best_metric:
            best_metric = current
            patience_counter = 0
            best_state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            LOGGER.info(
                f"New best {cfg.early_stop_metric}: {best_metric:.2f} at epoch {epoch+1}"
            )
            return False

        # No improvement
        patience_counter += 1
        LOGGER.info(f"No improvement for {patience_counter} eval steps.")

        if patience_counter >= cfg.patience:
            LOGGER.info(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best {cfg.early_stop_metric}: {best_metric:.2f}"
            )
            return True

        return False


    LOGGER.info(
        f"Starting training with config: "
        f"{ {k: v for k, v in asdict(cfg).items() if k != 'model_file'} }"
    )
    
    mode_str = "adv" if cfg.adv_train else "norm"
    
    # Choose learning rate based on architecture and training mode
    if cfg.arch == "wrn28x10":
        base_lr_nat = 0.10
        base_lr_adv = 0.01
    elif cfg.arch == "cnn":
        base_lr_nat = 0.01
        base_lr_adv = 0.005
    elif cfg.arch == "resnet18":
        # Paper training uses SGD 0.1 for natural and adversarial
        base_lr_nat = 0.10
        base_lr_adv = 0.10   # paper uses same LR for adv + nat
 
    else:
        raise ValueError(f"Unsupported architecture: {cfg.arch}")

    base_lr = base_lr_adv if cfg.adv_train else base_lr_nat

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
        )
    
    # Define warmup for first epochs
    warmup_epochs = 5
    total_epochs = cfg.num_epochs

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs)
    )
        
    best_metric = -float("inf")
    patience_counter = 0
    best_state = None

    logs: List[Dict] = []

    for epoch in range(cfg.num_epochs):
        _warmup_lr(epoch)
        epoch_start = time.time()

        # Train for one epoch
        avg_train_loss = train_one_epoch(
            net, train_loader, optimizer, criterion, cfg
        )
        epoch_time = time.time() - epoch_start

        # Evaluate periodically
        acc_nat = None
        acc_pgd_linf = None
        acc_pgd_l2 = None
        per_class_nat = None
        per_class_pgd_linf = None
        per_class_pgd_l2 = None

        if (epoch + 1) % cfg.eval_every == 0 or (epoch + 1) == cfg.num_epochs:
            metrics = evaluate_model(net, valid_loader, cfg)
            acc_nat = metrics["acc_nat"]
            acc_pgd_linf = metrics["acc_pgd_linf"]
            acc_pgd_l2 = metrics["acc_pgd_l2"]
            per_class_nat = metrics["per_class_nat"]
            per_class_pgd_linf = metrics["per_class_pgd_linf"]
            per_class_pgd_l2 = metrics["per_class_pgd_l2"]

            LOGGER.info(
                f"[Epoch {epoch+1:03d}] "
                f"train_loss={avg_train_loss:.4f} | "
                f"nat={acc_nat:.2f}% | "
                f"PGD-l∞={acc_pgd_linf:.2f}% | "
                f"PGD-l2={acc_pgd_l2:.2f}% | "
                f"time={epoch_time:.1f}s"
            )

        logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "epoch_time_sec": epoch_time,
                "acc_nat": acc_nat,
                "acc_pgd_linf": acc_pgd_linf,
                "acc_pgd_l2": acc_pgd_l2,
                "per_class_nat": per_class_nat,
                "per_class_pgd_linf": per_class_pgd_linf,
                "per_class_pgd_l2": per_class_pgd_l2
            }
        )

        # Test if early stopping is triggered
        if _is_early_stop(acc_nat, acc_pgd_linf, acc_pgd_l2, epoch):
            if best_state is not None:
                net.load_state_dict(best_state["model"])
            break

    # Save model
    os.makedirs(os.path.dirname(cfg.model_file), exist_ok=True)
    net.save(cfg.model_file)
    LOGGER.info(f"Model saved in {cfg.model_file}")

    # Save logs
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_filename = f"{cfg.arch}_training_{mode_str}_seed_{cfg.seed}.json"
    log_path = os.path.join(cfg.log_dir, log_filename)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    LOGGER.info(f"Training logs saved to {log_path}")

    return logs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        default=Net.model_file,
        help="Path to load/store model weights. "
             f"Default is '{Net.model_file}', which is used for testing."
    )
    parser.add_argument(
        "-f",
        "--force-train",
        action="store_true",
        help="Force training even if model file already exists."
    )
    parser.add_argument(
        "-e",
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training/validation."
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=1024,
        help="Number of examples reserved for validation."
    )
    parser.add_argument(
        "--adv-train",
        action="store_true",
        help="Enable PGD-linf adversarial training."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
    "--arch",
    type=str,
    default="resnet18",
    choices=["cnn", "wrn28x10", "resnet18"],
    help="Model architecture: cnn | wrn28x10 | resnet18"
)

    parser.add_argument(
    "--activation",
    type=str,
    default="gelu",
    choices=["relu", "gelu"],
    help="Activation function to use."
)


    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping during training."
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="balanced",
        choices=["acc_nat", "acc_pgd_linf", "acc_pgd_l2", "agg", "balanced"],
        help=(
            "Metric to monitor for early stopping. "
            "'agg' = acc_pgd_linf + acc_pgd_l2, "
            "'balanced' = 2 / (1/acc_pgd_linf + 1/acc_pgd_l2)"
        )
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of steps to wait for improvement before stopping."
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable data augmentation on the training set."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin_memory in DataLoaders (enabled by default on CUDA)."
    )
    parser.add_argument(
        "--eps-linf",
        type=float,
        default=8/255,
        help="PGD-linf ε (used for both training and testing)."
    )
    parser.add_argument(
        "--alpha-linf",
        type=float,
        default=2/255,
        help="PGD-linf step size (used for both training and testing)."
    )
    parser.add_argument(
        "--train-pgd-steps",
        type=int,
        default=10,
        help="Number of PGD steps during adversarial training."
    )
    parser.add_argument(
        "--test-pgd-steps",
        type=int,
        default=10,
        help="Number of PGD steps during adversarial evaluation."
    )
    parser.add_argument(
        "--eps-l2",
        type=float,
        default=1.0,
        help="PGD-l2 ε for testing."
    )
    parser.add_argument(
        "--alpha-l2",
        type=float,
        default=0.25,
        help="PGD-l2 step size for testing."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate on validation set every N epochs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save training logs."
    )

    return parser.parse_args()


def main() -> None:
    """Main function to train and evaluate the model."""
    args = parse_args()

    cfg = TrainingConfig(
        model_file=args.model_file,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        adv_train=args.adv_train,
        seed=args.seed,
        arch=args.arch,
        early_stop=not args.no_early_stop,
        early_stop_metric=args.early_stop_metric,
        patience=args.patience, 
        use_aug=not args.no_aug,
        num_workers=args.num_workers,
        pin_memory=(USE_CUDA and not args.no_pin_memory),
        train_eps_linf=args.eps_linf,
        train_alpha_linf=args.alpha_linf,
        train_pgd_steps=args.train_pgd_steps,
        test_eps_linf=args.eps_linf,
        test_alpha_linf=args.alpha_linf,
        test_eps_l2=args.eps_l2,
        test_alpha_l2=args.alpha_l2,
        test_pgd_steps=args.test_pgd_steps,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
        activation=args.activation,   
    )

    set_seed(cfg.seed)

    # net = Net().to(DEVICE)
    net = Net(arch=cfg.arch, activation=cfg.activation).to(DEVICE)


    # Build dataloaders
    train_loader, valid_loader = build_dataloaders(cfg)

    # Train the model if needed
    if not os.path.exists(cfg.model_file) or args.force_train:
        LOGGER.info(f"Training model -> {cfg.model_file}")
        train_model(net, train_loader, valid_loader, cfg)
    else:
        LOGGER.info(f"Model file '{cfg.model_file}' exists, skipping training.")

    # Evaluate the model
    LOGGER.info(f"Testing with model from '{cfg.model_file}'.")
    net.load(cfg.model_file)

    metrics = evaluate_model(net, valid_loader, cfg)
    agg = metrics["acc_pgd_linf"] + metrics["acc_pgd_l2"]

    LOGGER.info(f"Model natural accuracy   : {metrics['acc_nat']:.2f}")
    LOGGER.info(f"Model PGD-linf accuracy  : {metrics['acc_pgd_linf']:.2f}")
    LOGGER.info(f"Model PGD-l2  accuracy   : {metrics['acc_pgd_l2']:.2f}")
    LOGGER.info(f"Agg (PGD-linf + PGD-l2)  : {agg:.2f}")

    if cfg.model_file != Net.model_file:
        LOGGER.info(
            f"Warning: '{cfg.model_file}' is not the default model file; "
            f"it will not be the one used for automatic testing. "
            f"If this is your best model, rename/link it to '{Net.model_file}'."
        )


if __name__ == "__main__":
    main()