from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.attacks import pgd_linf_attack, pgd_l2_attack
from scripts.configs import TrainingConfig, DEVICE

def eval_accuracy(
    net: nn.Module,
    loader: DataLoader,
    num_classes: int = 10,
    attack_fn=None,
    attack_kwargs: Optional[Dict] = None
) -> Tuple[float, List[float]]:
    """
    Evaluate accuracy on a loader, optionally under an attack.

    Returns:
        overall_acc: float in [0, 100]
        per_class_acc: list of length num_classes, each in [0, 100]
    """
    net.eval()
    correct_per_class = torch.zeros(num_classes, device=DEVICE)
    total_per_class = torch.zeros(num_classes, device=DEVICE)

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if attack_fn is not None:
            if attack_kwargs is None:
                attack_kwargs = {}
            images = attack_fn(net, images, labels, **attack_kwargs)

        outputs = net(images)
        preds = outputs.argmax(dim=1)

        for c in range(num_classes):
            mask = (labels == c)
            if mask.any():
                total_per_class[c] += mask.sum()
                correct_per_class[c] += (preds[mask] == c).sum()

    acc_per_class = torch.where(
        total_per_class > 0,
        correct_per_class / total_per_class * 100.0,
        torch.zeros_like(total_per_class),
    )

    overall_acc = (correct_per_class.sum() / total_per_class.sum() * 100.0).item()
    return overall_acc, acc_per_class.cpu().tolist()


def evaluate_model(
    net: nn.Module,
    loader: DataLoader,
    cfg: TrainingConfig
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute natural and adversarial accuracies (overall + per-class)
    on a given loader.
    """
    # Natural
    acc_nat, per_class_nat = eval_accuracy(net, loader, num_classes=10)

    # PGD-linf
    acc_pgd_linf, per_class_pgd_linf = eval_accuracy(
        net,
        loader,
        num_classes=10,
        attack_fn=pgd_linf_attack,
        attack_kwargs=dict(
            epsilon=cfg.test_eps_linf,
            alpha=cfg.test_alpha_linf,
            num_iter=cfg.test_pgd_steps,
        ),
    )

    # PGD-l2
    acc_pgd_l2, per_class_pgd_l2 = eval_accuracy(
        net,
        loader,
        num_classes=10,
        attack_fn=pgd_l2_attack,
        attack_kwargs=dict(
            epsilon=cfg.test_eps_l2,
            alpha=cfg.test_alpha_l2,
            num_iter=cfg.test_pgd_steps,
        ),
    )

    return {
        "acc_nat": acc_nat,
        "acc_pgd_linf": acc_pgd_linf,
        "acc_pgd_l2": acc_pgd_l2,
        "per_class_nat": per_class_nat,
        "per_class_pgd_linf": per_class_pgd_linf,
        "per_class_pgd_l2": per_class_pgd_l2
    }
