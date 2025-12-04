from dataclasses import dataclass
import logging
import sys
import torch


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[{asctime}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)


# Configure device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
LOGGER.info(
    f"use_cuda={USE_CUDA}, cuda_device_count={torch.cuda.device_count()}"
)


@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    # Model parameters
    model_file: str
    num_epochs: int = 200
    batch_size: int = 256
    valid_size: int = 1024
    adv_train: bool = False
    seed: int = 0
    arch: str = "wrn28x10"
    # arch: str = "cnn"
    early_stop: bool = True
    early_stop_metric: str = "balanced"
    patience: int = 5 
    use_aug: bool = True

    arch: str = "resnet18"          # cnn, wrn28x10, or resnet18
    activation: str = "gelu" 
    
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = USE_CUDA

    # PGD for training (inner maximization)
    train_eps_linf: float = 8 / 255
    train_alpha_linf: float = 0.0078
    train_pgd_steps: int = 10

    # PGD for evaluation
    test_eps_linf: float = 8 / 255
    test_alpha_linf: float = 2 / 255
    test_eps_l2: float = 1.0
    test_alpha_l2: float = 0.25
    test_pgd_steps: int = 10

    # Logging / eval
    eval_every: int = 1
    log_dir: str = "logs"


