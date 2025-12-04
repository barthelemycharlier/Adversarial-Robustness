import os
import torch
import torch.nn as nn
import argparse

from model import Net
from scripts.configs import TrainingConfig, DEVICE, LOGGER
from scripts.data import build_dataloaders
from scripts.evaluate import evaluate_model


class MeanSparse(nn.Module):
    def __init__(self, in_planes: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(in_planes))
        self.register_buffer("running_var", torch.zeros(in_planes))
        self.register_buffer("threshold", torch.tensor(0.0))

        self.register_buffer("flag_update_statistics", torch.tensor(0))
        self.register_buffer("batch_num", torch.tensor(0.0))

    def forward(self, x):
        if self.flag_update_statistics:
            m = torch.mean(x.detach(), dim=(0, 2, 3))
            v = torch.var(x.detach(), dim=(0, 2, 3), unbiased=False)
            self.running_mean += m / self.batch_num
            self.running_var += v / self.batch_num

        if self.threshold.item() == 0:
            return x

        bias = self.running_mean.view(1, -1, 1, 1)
        crop = self.threshold * torch.sqrt(self.running_var + 1e-8).view(1, -1, 1, 1)
        diff = x - bias

        return torch.where(torch.abs(diff) < crop, bias, x)


def add_meansparse_generic(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.Sequential(module, MeanSparse(module.num_features)))
        add_meansparse_generic(module)


def add_meansparse_to_wrn(backbone):
    from scripts.models import WideResNetBackbone, WideBasic
    assert isinstance(backbone, WideResNetBackbone)

    for m in backbone.modules():
        if isinstance(m, WideBasic):
            if isinstance(m.bn1, nn.BatchNorm2d):
                m.bn1 = nn.Sequential(m.bn1, MeanSparse(m.bn1.num_features))
            if isinstance(m.bn2, nn.BatchNorm2d):
                m.bn2 = nn.Sequential(m.bn2, MeanSparse(m.bn2.num_features))

    if isinstance(backbone.bn1, nn.BatchNorm2d):
        backbone.bn1 = nn.Sequential(backbone.bn1, MeanSparse(backbone.bn1.num_features))


def load_model_weights(net, path):
    sd = torch.load(path, map_location=DEVICE)

    if any(k.startswith("backbone.") for k in sd.keys()):
        LOGGER.info("[Loader] Detected WRN checkpoint.")
        net.load_state_dict(sd)
        return

    backbone_state = net.backbone.state_dict()
    if set(sd.keys()) == set(backbone_state.keys()):
        LOGGER.info("[Loader] Detected CNN checkpoint.")
        net.backbone.load_state_dict(sd)
        return

    raise RuntimeError("Checkpoint incompatible with architecture.")


def inject_meansparse(net, arch):
    if arch == "cnn":
        LOGGER.info("[MeanSparse] Inserting MeanSparse into CNN.")
        add_meansparse_generic(net.backbone)
    elif arch == "wrn28x10":
        LOGGER.info("[MeanSparse] Inserting MeanSparse into WRN28x10.")
        add_meansparse_to_wrn(net.backbone)
    else:
        raise ValueError("Unknown architecture.")


def compute_meansparse_stats(model, loader, device=DEVICE):
    model.eval()
    num_batches = len(loader)

    for m in model.modules():
        if isinstance(m, MeanSparse):
            m.flag_update_statistics.fill_(1)
            m.batch_num.fill_(float(num_batches))
            m.running_mean.zero_()
            m.running_var.zero_()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)

    for m in model.modules():
        if isinstance(m, MeanSparse):
            m.flag_update_statistics.fill_(0)


def set_global_threshold(model, alpha):
    for m in model.modules():
        if isinstance(m, MeanSparse):
            m.threshold.fill_(alpha)


def sweep_alphas(model, loader, cfg, alphas):
    base = evaluate_model(model, loader, cfg)
    base_nat  = base["acc_nat"]
    base_inf  = base["acc_pgd_linf"]
    base_l2   = base["acc_pgd_l2"]

    best_alpha = None
    best_rob = -1

    LOGGER.info(
        f"[MeanSparse] Baseline nat={base_nat:.2f}, "
        f"L_inf={base_inf:.2f}, L2={base_l2:.2f}"
    )

    alpha_stats = []

    for a in alphas:
        set_global_threshold(model, a)
        m = evaluate_model(model, loader, cfg)

        nat     = m["acc_nat"]
        rob_inf = m["acc_pgd_linf"]
        rob_l2  = m["acc_pgd_l2"]

        LOGGER.info(
            f"[alpha={a:.3f}] nat={nat:.2f}, "
            f"rob_Linf={rob_inf:.2f}, rob_L2={rob_l2:.2f}"
        )

        alpha_stats.append((a, nat, rob_inf, rob_l2))

        if nat >= base_nat - 0.5 and rob_inf > best_rob:
            best_rob = rob_inf
            best_alpha = a

    if best_alpha is None:
        LOGGER.warning("No alpha kept clean accuracy → fallback to max robust (L_inf).")
        best_alpha = max(
            alphas,
            key=lambda a: evaluate_model(model, loader, cfg)["acc_pgd_linf"]
        )


    print("\n=== MeanSparse Alpha Sweep Summary ===")
    print("alpha      nat    L_inf    L2")
    print("---------------------------------------")
    print(f"baseline  {base_nat:5.1f}  {base_inf:6.1f}  {base_l2:6.1f}")
    print("---------------------------------------")
    for a, nat, r_inf, r_l2 in alpha_stats:
        print(f"{a:8.3f}  {nat:5.1f}  {r_inf:6.1f}  {r_l2:6.1f}")
    print("---------------------------------------")
    print(f"Selected alpha = {best_alpha:.3f}\n")

    LOGGER.info(f"[MeanSparse] Selected alpha={best_alpha:.3f}")
    return best_alpha



def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", type=str, required=True)
    p.add_argument("--arch", type=str, choices=["cnn", "wrn28x10"], required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.03, 0.05, 0.08])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--valid-size", type=int, default=1024)
    return p.parse_args()


def main():
    args = get_args()

    cfg = TrainingConfig(
        model_file=args.model_file,
        arch=args.arch,
        adv_train=True,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        log_dir="logs"
    )

    _, valid_loader = build_dataloaders(cfg)

    net = Net(arch=cfg.arch).to(DEVICE)

    LOGGER.info(f"[Load] {args.model_file}")
    load_model_weights(net, args.model_file)

    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    inject_meansparse(net, args.arch)
    for m in net.modules():
        if isinstance(m, MeanSparse):
            m.running_mean = m.running_mean.to(DEVICE)
            m.running_var = m.running_var.to(DEVICE)
            m.threshold = m.threshold.to(DEVICE)
            m.flag_update_statistics = m.flag_update_statistics.to(DEVICE)
            m.batch_num = m.batch_num.to(DEVICE)


    LOGGER.info("[MeanSparse] Computing μ/σ...")
    compute_meansparse_stats(net, _, DEVICE)

    LOGGER.info(f"[MeanSparse] Sweeping alphas: {args.alphas}")
    best_alpha = sweep_alphas(net, valid_loader, cfg, args.alphas)

    out = f"models/{args.arch}_meansparse_seed_{os.path.basename(args.model_file)}_alpha{best_alpha:.3f}.pth"
    torch.save(net.state_dict(), out)
    LOGGER.info(f"[MeanSparse] Saved → {out}")


if __name__ == "__main__":
    main()
