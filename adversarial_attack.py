"""
Adversarial Noise Experiment
============================
Demonstrates how small, carefully crafted perturbations (adversarial noise)
can fool a neural network into misclassifying an image.

Methods implemented:
  - FGSM  (Fast Gradient Sign Method)  — single-step attack
  - PGD   (Projected Gradient Descent)  — iterative attack (stronger)
  - MI-FGSM (Momentum Iterative FGSM)  — better transferability
  - Ensemble MI-DI-TI-FGSM            — best transferability across models

Usage:
  python adversarial_attack.py --image couch.jpg --target 281
  python adversarial_attack.py --image couch.jpg --target 281 --method pgd

ImageNet class indices reference:
  281 = tabby cat
  283 = Persian cat
  285 = Egyptian cat
  820 = couch / sofa
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# ── ImageNet normalisation stats ──────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ── A small subset of ImageNet labels (full list has 1000 entries) ────────────
# We'll load them dynamically if possible; otherwise fall back to a minimal map.
LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)


def load_imagenet_labels() -> list[str]:
    """Try to fetch human-readable ImageNet labels; fall back to indices."""
    try:
        import urllib.request
        with urllib.request.urlopen(LABELS_URL, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return [str(i) for i in range(1000)]


def load_image(path: str, size: int = 224) -> torch.Tensor:
    """Load an image and return a normalised 1×3×H×W tensor."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # -> [0, 1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # 1×3×224×224


def normalise(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * IMAGENET_STD + IMAGENET_MEAN


def predict(model: torch.nn.Module, img_tensor: torch.Tensor,
            labels: list[str]) -> tuple[int, str, float]:
    """Run inference and return (class_idx, label, confidence%)."""
    with torch.no_grad():
        logits = model(normalise(img_tensor))
    probs = F.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    idx = idx.item()
    return idx, labels[idx], conf.item() * 100


# ── Attack methods ────────────────────────────────────────────────────────────

def fgsm_targeted(model: torch.nn.Module, img: torch.Tensor,
                  target_class: int, epsilon: float = 0.03) -> torch.Tensor:
    """
    Fast Gradient Sign Method (targeted).

    Perturbs the image in one step to *minimise* the loss for the target class
    (i.e. make the model more confident about the target).
    """
    perturbed = img.clone().requires_grad_(True)
    logits = model(normalise(perturbed))
    loss = F.cross_entropy(logits, torch.tensor([target_class]))

    model.zero_grad()
    loss.backward()

    # Move *against* the gradient to minimise loss for target class
    adv = perturbed - epsilon * perturbed.grad.sign()
    return adv.detach().clamp(0, 1)


def pgd_targeted(model: torch.nn.Module, img: torch.Tensor,
                 target_class: int, epsilon: float = 0.03,
                 alpha: float = 0.005, steps: int = 40,
                 progress_callback=None) -> torch.Tensor:
    """
    Projected Gradient Descent (targeted, iterative).

    Stronger than FGSM — takes many small steps and projects back
    into the epsilon-ball around the original image.
    """
    adv = img.clone()
    for step in range(steps):
        if progress_callback:
            progress_callback(step, steps)
        adv.requires_grad_(True)
        logits = model(normalise(adv))
        loss = F.cross_entropy(logits, torch.tensor([target_class]))

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            # Project back into ε-ball
            delta = torch.clamp(adv - img, -epsilon, epsilon)
            adv = torch.clamp(img + delta, 0, 1)

    return adv.detach()


# ── Transferability-enhancing techniques ──────────────────────────────────────

def _input_diversity(x: torch.Tensor, prob: float = 0.7,
                     low: int = 200, high: int = 224) -> torch.Tensor:
    """DI-FGSM: random resize + pad to improve transfer (Xie et al., 2019)."""
    if torch.rand(1).item() > prob:
        return x
    rnd = torch.randint(low, high, (1,)).item()
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear',
                             align_corners=False)
    pad_h = high - rnd
    pad_w = high - rnd
    pad_top = torch.randint(0, pad_h + 1, (1,)).item()
    pad_left = torch.randint(0, pad_w + 1, (1,)).item()
    padded = F.pad(rescaled,
                   (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))
    return padded


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Create 2D Gaussian kernel for TI-FGSM (Dong et al., 2019)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).repeat(3, 1, 1, 1)


def ensemble_mi_di_ti_fgsm(
    model_list: list[torch.nn.Module],
    img: torch.Tensor,
    target_class: int,
    epsilon: float = 0.06,
    alpha: float = 0.004,
    steps: int = 100,
    momentum: float = 1.0,
    di_prob: float = 0.7,
    ti_kernel_size: int = 5,
    progress_callback=None,
) -> torch.Tensor:
    """
    Ensemble MI-DI-TI-FGSM (targeted).

    Combines multiple techniques for maximum transferability:
      - Ensemble: average gradients from multiple models
      - MI: momentum to escape poor local optima (Dong et al., 2018)
      - DI: input diversity via random resize+pad (Xie et al., 2019)
      - TI: translation-invariant via Gaussian-smoothed gradients (Dong et al., 2019)
    """
    adv = img.clone()
    grad_momentum = torch.zeros_like(img)

    # Gaussian kernel for translation-invariant smoothing
    ti_kernel = _gaussian_kernel(ti_kernel_size)

    target_tensor = torch.tensor([target_class])

    for step in range(steps):
        if progress_callback:
            progress_callback(step, steps)
        adv.requires_grad_(True)
        total_grad = torch.zeros_like(img)

        for model in model_list:
            # Apply input diversity
            adv_di = _input_diversity(adv, prob=di_prob)
            logits = model(normalise(adv_di))
            loss = F.cross_entropy(logits, target_tensor)
            model.zero_grad()
            loss.backward()
            total_grad += adv.grad.data.clone()
            adv.grad.data.zero_()

        # Average over ensemble
        total_grad /= len(model_list)

        # TI: smooth gradient with Gaussian kernel
        total_grad = F.conv2d(total_grad, ti_kernel, padding=ti_kernel_size // 2,
                              groups=3)

        # MI: update momentum
        grad_norm = total_grad / (total_grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)
        grad_momentum = momentum * grad_momentum + grad_norm

        with torch.no_grad():
            # Targeted: move against gradient to minimise loss for target class
            adv = adv - alpha * grad_momentum.sign()
            delta = torch.clamp(adv - img, -epsilon, epsilon)
            adv = torch.clamp(img + delta, 0, 1)

    return adv.detach()


# ── Visualisation ─────────────────────────────────────────────────────────────

def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert 1×3×H×W tensor [0,1] to H×W×3 numpy array."""
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def visualise(original: torch.Tensor, adversarial: torch.Tensor,
              orig_label: str, orig_conf: float,
              adv_label: str, adv_conf: float,
              save_path: str = "output/result.png",
              noise_scale: float = 10.0) -> None:
    """
    Save a 3-panel figure:
      [Original image]  [Amplified noise]  [Adversarial image]
    """
    noise = adversarial - original

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(to_numpy_image(original))
    axes[0].set_title(f"Original\n{orig_label} ({orig_conf:.1f}%)", fontsize=12)
    axes[0].axis("off")

    # Amplify noise for visibility; shift to [0,1] range
    noise_vis = to_numpy_image(noise) * noise_scale + 0.5
    noise_vis = np.clip(noise_vis, 0, 1)
    axes[1].imshow(noise_vis)
    axes[1].set_title(f"Adversarial Noise (×{noise_scale:.0f})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(to_numpy_image(adversarial))
    axes[2].set_title(f"Adversarial\n{adv_label} ({adv_conf:.1f}%)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved visualisation → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial Noise Experiment: fool a neural network "
                    "into misclassifying an image."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image. If omitted, a synthetic test image is generated."
    )
    parser.add_argument(
        "--target", type=int, default=281,
        help="Target ImageNet class index (default: 281 = tabby cat)."
    )
    parser.add_argument(
        "--method", choices=["fgsm", "pgd"], default="pgd",
        help="Attack method: fgsm (fast, single step) or pgd (stronger, iterative)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.03,
        help="Maximum perturbation magnitude (L∞ norm). Default: 0.03"
    )
    parser.add_argument(
        "--steps", type=int, default=40,
        help="Number of PGD iterations (only for --method pgd). Default: 40"
    )
    parser.add_argument(
        "--output", type=str, default="output/result.png",
        help="Path to save the output visualisation."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Adversarial Noise Experiment")
    print("=" * 60)

    # 1. Load labels
    print("\n[1/5] Loading ImageNet labels …")
    labels = load_imagenet_labels()

    # 2. Load model
    print("[2/5] Loading pre-trained ResNet-50 …")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    # 3. Load / generate image
    if args.image and Path(args.image).exists():
        print(f"[3/5] Loading image: {args.image}")
        img = load_image(args.image)
    else:
        print("[3/5] No image provided — generating synthetic test image …")
        img = generate_test_image()

    # 4. Classify original
    orig_idx, orig_label, orig_conf = predict(model, img, labels)
    print(f"       Original prediction: {orig_label} (idx={orig_idx}, "
          f"conf={orig_conf:.1f}%)")
    print(f"       Target class:        {labels[args.target]} (idx={args.target})")

    # 5. Run attack
    method_name = args.method.upper()
    print(f"\n[4/5] Running {method_name} targeted attack "
          f"(ε={args.epsilon}, target={args.target}) …")

    if args.method == "fgsm":
        adv_img = fgsm_targeted(model, img, args.target, args.epsilon)
    else:
        adv_img = pgd_targeted(model, img, args.target, args.epsilon,
                               steps=args.steps)

    adv_idx, adv_label, adv_conf = predict(model, adv_img, labels)
    print(f"       After attack:        {adv_label} (idx={adv_idx}, "
          f"conf={adv_conf:.1f}%)")

    success = adv_idx == args.target
    if success:
        print(f"\n  ★ Attack SUCCEEDED — model now sees '{adv_label}' "
              f"with {adv_conf:.1f}% confidence")
    else:
        print(f"\n  ✗ Attack did not reach target class. "
              f"Try increasing --epsilon or --steps.")

    # 6. Visualise
    print(f"\n[5/5] Generating visualisation …")
    visualise(img, adv_img, orig_label, orig_conf,
              adv_label, adv_conf, save_path=args.output)

    # Noise statistics
    noise = adv_img - img
    l_inf = noise.abs().max().item()
    l_2 = noise.norm(2).item()
    print(f"\n  Noise stats:  L∞ = {l_inf:.4f},  L₂ = {l_2:.4f}")
    print(f"  Max pixel change: {l_inf * 255:.1f} / 255")
    print()


def generate_test_image(size: int = 224) -> torch.Tensor:
    """
    Generate a simple synthetic image (coloured rectangles)
    that pretrained models can still classify — useful for quick testing
    without needing an actual photo.
    """
    img = torch.zeros(1, 3, size, size)
    # brown-ish base (couch-like colour)
    img[:, 0, :, :] = 0.45
    img[:, 1, :, :] = 0.30
    img[:, 2, :, :] = 0.20
    # add some texture
    torch.manual_seed(42)
    img += torch.randn_like(img) * 0.05
    img.clamp_(0, 1)
    return img


if __name__ == "__main__":
    main()
