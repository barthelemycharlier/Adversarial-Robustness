import torch
import torch.nn.functional as F


def pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8/255,
    alpha: float = 2/255,
    num_iter: int = 10
):
    """
    Projected Gradient Descent Attack (with projection onto the l∞ ball).

    Objective:
        Solve   max_{||δ||∞ ≤ ε}   L(model(x + δ), y)

    Algorithm:
        1. Start from a random point inside the l∞ ball:
               x_adv₀ = x + Uniform(-ε, ε)
        2. For t = 1,...,T:
               x_adv_{t+1} = x_adv_t + alpha * sign( ∇_x L(model(x_adv_t), y) )
               Project back to l∞ ball:
                   x_adv_{t+1} = x + clamp(x_adv_{t+1} - x, -ε, ε)

    Parameters:
        epsilon  : l_inf radius of allowed perturbation
        alpha    :   step size
        num_iter : number of gradient ascent steps

    Returns:
        x_adv final adversarial batch
    """

    # Set model to evaluation mode
    model.eval()
    # Store original inputs
    x_orig = x.detach()

    # Random start inside the ε-box
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Iterative gradient ascent
    for t in range(num_iter):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.nll_loss(outputs, y)
        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.sign()

        # Gradient ascent step
        x_adv = x_adv + alpha * grad_sign

        # Project back to ε-L∞ ball around the clean input
        eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + eta, 0.0, 1.0).detach()

    return x_adv

def pgd_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1.0,
    alpha: float = 0.25,
    num_iter: int = 10
):
    """
    Projected Gradient Descent Attack (with projection onto the l2 ball).

    Objective:
        Solve   max_{||δ||₂ ≤ ε}   L(model(x + δ), y)

    Key idea:
        Unlike l∞, l2 does NOT use the sign of the gradient.
        Instead, δ is updated in the direction of the gradient normalized to
        unit l2 length:

            δ ← δ + alpha * ( ∇_x L / ||∇_x L||₂ )

        Then projected onto the l2 ball:
            δ ← ε * δ / max(ε, ||δ||₂)

    Geometric interpretation:
        Find the perturbation with bounded energy (Euclidean norm) that makes
        the classifier maximally confused.

    Parameters:
        epsilon  : l2 radius of allowed perturbation
        alpha    :   gradient ascent step size
        num_iter : number of steps

    Returns:
        x_adv = x + δ
    """

    # Set model to evaluation mode
    model.eval()
    # Store original inputs
    x_orig = x.detach()


    # Start from δ = 0 (can also randomize inside the ball)
    delta = torch.zeros_like(x_orig)

    for t in range(num_iter):

        # Compute adversarial example for gradient computation
        x_adv = (x_orig + delta).detach().requires_grad_(True)

        outputs = model(x_adv)
        loss = F.nll_loss(outputs, y)
        model.zero_grad()
        loss.backward()

        # Get gradient
        grad = x_adv.grad

        # Flatten for norm computation
        grad_flat = grad.view(grad.size(0), -1)

        # Compute l2 norms
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_norm = torch.clamp(grad_norm, min=1e-8)

        # Normalize gradient to unit l2 vector
        normalized_grad = (grad_flat / grad_norm).view_as(grad)

        # Take a step in the direction of steepest increase of loss
        delta = delta + alpha * normalized_grad

        # Project delta onto l2 ball of radius ε
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)

        factor = torch.min(torch.ones_like(delta_norm), epsilon / delta_norm)
        delta = (delta_flat * factor).view_as(delta)

        # x_adv must remain a valid image
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        delta = (x_adv - x_orig).detach()

    return x_orig + delta