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
import io
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


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ── MPS compatibility: adaptive_avg_pool2d doesn't support all sizes ─────────
# PyTorch issue #96056 — fallback to CPU for the pool op (tensor is tiny, no perf hit)

if DEVICE.type == "mps":
    _orig_adaptive_avg_pool2d = F.adaptive_avg_pool2d

    def _mps_safe_adaptive_avg_pool2d(input, output_size):
        if input.device.type == "mps":
            return _orig_adaptive_avg_pool2d(
                input.cpu(), output_size,
            ).to("mps")
        return _orig_adaptive_avg_pool2d(input, output_size)

    F.adaptive_avg_pool2d = _mps_safe_adaptive_avg_pool2d


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
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return (tensor - mean) / std


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    return tensor * std + mean


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
    dev = next(model.parameters()).device
    img_d = img.to(dev)
    perturbed = img_d.clone().requires_grad_(True)
    logits = model(normalise(perturbed))
    loss = F.cross_entropy(logits, torch.tensor([target_class], device=dev))

    model.zero_grad()
    loss.backward()

    adv = perturbed - epsilon * perturbed.grad.sign()
    return adv.detach().clamp(0, 1).cpu()


def pgd_targeted(model: torch.nn.Module, img: torch.Tensor,
                 target_class: int, epsilon: float = 0.03,
                 alpha: float = 0.005, steps: int = 40,
                 progress_callback=None) -> torch.Tensor:
    """
    Projected Gradient Descent (targeted, iterative).

    Stronger than FGSM — takes many small steps and projects back
    into the epsilon-ball around the original image.
    """
    dev = next(model.parameters()).device
    img_d = img.to(dev)
    adv = img_d.clone()
    target_t = torch.tensor([target_class], device=dev)
    for step in range(steps):
        if progress_callback:
            progress_callback(step, steps)
        adv.requires_grad_(True)
        logits = model(normalise(adv))
        loss = F.cross_entropy(logits, target_t)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            delta = torch.clamp(adv - img_d, -epsilon, epsilon)
            adv = torch.clamp(img_d + delta, 0, 1)

    return adv.detach().cpu()


# ── Transferability-enhancing techniques ──────────────────────────────────────

def _input_diversity(x: torch.Tensor, prob: float = 0.7,
                     low: int = 200, high: int = 224) -> torch.Tensor:
    """DI-FGSM: random resize + pad to improve transfer (Xie et al., 2019)."""
    if torch.rand(1).item() > prob:
        return x
    # Sizes must be divisible by 7 for adaptive_avg_pool2d on MPS (Apple Silicon)
    valid_sizes = [s for s in range(low, high) if s % 7 == 0]
    if not valid_sizes:
        valid_sizes = [s for s in range(low - 7, high + 7) if s % 7 == 0 and s >= 56]
    rnd = valid_sizes[torch.randint(0, len(valid_sizes), (1,)).item()]
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear',
                             align_corners=False)
    pad_h = high - rnd
    pad_w = high - rnd
    pad_top = torch.randint(0, max(1, pad_h + 1), (1,)).item()
    pad_left = torch.randint(0, max(1, pad_w + 1), (1,)).item()
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


# ── Compression-robust augmentations ─────────────────────────────────────────

def _jpeg_compress_ste(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Apply real JPEG compression with Straight-Through Estimator.
    Forward pass: actual JPEG compression (non-differentiable).
    Backward pass: gradient passes straight through (identity).
    """
    from PIL import Image as _PILImage

    with torch.no_grad():
        dev = x.device
        img_np = x.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = _PILImage.fromarray(img_np)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = _PILImage.open(buf).convert("RGB")

        comp_np = np.array(compressed).astype(np.float32) / 255.0
        comp_tensor = torch.from_numpy(comp_np).permute(2, 0, 1).unsqueeze(0).to(dev)

    # STE: use compressed result for forward, pass gradient straight through
    return x + (comp_tensor - x).detach()


def _gaussian_blur_tensor(x: torch.Tensor, kernel_size: int = 5,
                          sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to a tensor (differentiable)."""
    k = _gaussian_kernel(kernel_size, sigma).to(x.device)
    return F.conv2d(x, k, padding=kernel_size // 2, groups=3)


def _random_resize_ste(x: torch.Tensor, min_scale: float = 0.6,
                       max_scale: float = 0.9) -> torch.Tensor:
    """Random downscale then upscale to simulate re-encoding resolution loss."""
    _, _, h, w = x.shape
    scale = min_scale + torch.rand(1).item() * (max_scale - min_scale)
    sh, sw = int(h * scale), int(w * scale)
    # Ensure sizes divisible by 7 for MPS compatibility
    sh = max(7, (sh // 7) * 7)
    sw = max(7, (sw // 7) * 7)
    with torch.no_grad():
        down = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=False)
        up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
    return x + (up - x).detach()


def _social_media_augment(x: torch.Tensor, jpeg_range: tuple = (60, 95),
                          blur_prob: float = 0.3,
                          resize_prob: float = 0.3) -> torch.Tensor:
    """
    Random augmentation simulating social media upload pipeline:
    - JPEG compression (always, random quality 60-95)
    - Gaussian blur (30% chance, sigma 0.5-1.5)
    - Resize down+up (30% chance, scale 0.6-0.9)

    All use STE so gradients flow back for adversarial optimization.
    """
    # Always apply JPEG — this is what kills noise in practice
    q = torch.randint(jpeg_range[0], jpeg_range[1] + 1, (1,)).item()
    x = _jpeg_compress_ste(x, quality=q)

    # Random blur
    if torch.rand(1).item() < blur_prob:
        sigma = 0.5 + torch.rand(1).item() * 1.0
        x = _gaussian_blur_tensor(x, kernel_size=5, sigma=sigma)

    # Random resize
    if torch.rand(1).item() < resize_prob:
        x = _random_resize_ste(x)

    return x


# ── Face-aware attack helpers ─────────────────────────────────────────────────

def _crop_face_tensor(x: torch.Tensor, box: tuple,
                      target_size: int = 160) -> torch.Tensor:
    """Differentiable face crop + resize. box = (x1, y1, x2, y2) in pixels."""
    x1, y1, x2, y2 = [int(c) for c in box]
    h, w = x.shape[2], x.shape[3]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return F.interpolate(x, size=(target_size, target_size),
                             mode='bilinear', align_corners=False)
    crop = x[:, :, y1:y2, x1:x2]
    return F.interpolate(crop, size=(target_size, target_size),
                         mode='bilinear', align_corners=False)


def _apply_region_epsilon(mask: torch.Tensor, box: tuple,
                          eps: float, margin: float = 0.2) -> None:
    """Set epsilon in expanded box region (only increases, never decreases)."""
    x1, y1, x2, y2 = [int(c) for c in box]
    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
    mh, mw = int(bh * margin), int(bw * margin)
    y1e = max(0, y1 - mh)
    x1e = max(0, x1 - mw)
    y2e = min(mask.shape[2], y2 + mh)
    x2e = min(mask.shape[3], x2 + mw)
    mask[:, :, y1e:y2e, x1e:x2e] = torch.clamp(
        mask[:, :, y1e:y2e, x1e:x2e], min=eps,
    )


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
    init_delta: torch.Tensor | None = None,
    cancel_flag: list | None = None,
    compression_robust: bool = False,
    face_model: torch.nn.Module | None = None,
    face_box: tuple | None = None,
    face_weight: float = 1.0,
    text_boxes: list[tuple] | None = None,
) -> torch.Tensor:
    """
    Ensemble MI-DI-TI-FGSM (targeted).

    Combines multiple techniques for maximum transferability:
      - Ensemble: average gradients from multiple models
      - MI: momentum to escape poor local optima (Dong et al., 2018)
      - DI: input diversity via random resize+pad (Xie et al., 2019)
      - TI: translation-invariant via Gaussian-smoothed gradients (Dong et al., 2019)

    When compression_robust=True, additionally:
      - Simulates JPEG compression (quality 60-95) at each step (STE)
      - Random Gaussian blur and resize augmentation
      - Forces noise into low-frequency bands that survive social media recompression
      This makes the attack effective even after TikTok re-encodes the video/image.

    When face_model + face_box are provided (celebrity mode):
      - Adds face embedding loss to make faces unrecognizable
      - Uses per-pixel epsilon: stronger noise (2.5x) in face region

    When text_boxes are provided (OCR protection):
      - Uses per-pixel epsilon: stronger noise (3x) in text regions
      - Breaks OCR readability of numbers/text (bank balances, payouts)

    If init_delta is provided, start from img+init_delta (warm-start).
    Pass cancel_flag=[False] — set cancel_flag[0]=True to abort early.
    """
    dev = next(model_list[0].parameters()).device
    img_d = img.to(dev)

    if init_delta is not None:
        adv = torch.clamp(img_d + init_delta.to(dev), 0, 1)
    else:
        adv = img_d.clone()
    grad_momentum = torch.zeros_like(img_d)

    ti_kernel = _gaussian_kernel(ti_kernel_size).to(dev)
    target_tensor = torch.tensor([target_class], device=dev)

    # For compression-robust mode: low-frequency noise enforcement kernel
    if compression_robust:
        lf_kernel = _gaussian_kernel(7, sigma=1.5).to(dev)

    # Per-pixel epsilon mask (face + text regions get stronger noise)
    eps_mask = None
    has_face = face_box is not None
    has_text = text_boxes is not None and len(text_boxes) > 0
    if has_face or has_text:
        eps_mask = torch.full(img_d.shape, epsilon, device=dev)
        if has_face:
            face_eps = min(epsilon * 2.5, 0.20)
            _apply_region_epsilon(eps_mask, face_box, face_eps, margin=0.25)
        if has_text:
            text_eps = min(epsilon * 3.0, 0.25)
            for tbox in text_boxes:
                _apply_region_epsilon(eps_mask, tbox, text_eps, margin=0.15)

    # Face embedding attack setup
    orig_face_emb = None
    if face_model is not None and face_box is not None:
        with torch.no_grad():
            orig_crop = _crop_face_tensor(img_d, face_box, 160)
            orig_face_emb = face_model(orig_crop * 2 - 1).detach()

    for step in range(steps):
        if cancel_flag and cancel_flag[0]:
            break
        if progress_callback:
            progress_callback(step, steps)
        adv.requires_grad_(True)
        total_grad = torch.zeros_like(img_d)

        for model in model_list:
            # Standard DI augmentation
            adv_aug = _input_diversity(adv, prob=di_prob)

            # Compression-robust: simulate social media pipeline
            if compression_robust:
                adv_aug = _social_media_augment(adv_aug)

            logits = model(normalise(adv_aug))
            loss = F.cross_entropy(logits, target_tensor)
            model.zero_grad()
            loss.backward()
            total_grad += adv.grad.data.clone()
            adv.grad.data.zero_()

        # Face embedding loss: push face embedding away from original
        if face_model is not None and orig_face_emb is not None:
            face_crop = _crop_face_tensor(adv, face_box, 160)
            adv_emb = face_model(face_crop * 2 - 1)
            f_loss = F.cosine_similarity(adv_emb, orig_face_emb, dim=1).mean()
            f_loss.backward()
            total_grad += adv.grad.data.clone() * face_weight * len(model_list)
            adv.grad.data.zero_()

        total_grad /= len(model_list)

        # TI: smooth gradient
        total_grad = F.conv2d(total_grad, ti_kernel, padding=ti_kernel_size // 2,
                              groups=3)

        # MI: update momentum
        grad_norm = total_grad / (total_grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)
        grad_momentum = momentum * grad_momentum + grad_norm

        with torch.no_grad():
            adv = adv - alpha * grad_momentum.sign()
            delta = adv - img_d

            # Compression-robust: force noise into low frequencies
            if compression_robust:
                delta = F.conv2d(delta, lf_kernel, padding=3, groups=3)

            # Project into epsilon ball (per-pixel when face-aware)
            if eps_mask is not None:
                delta = torch.clamp(delta, -eps_mask, eps_mask)
            else:
                delta = torch.clamp(delta, -epsilon, epsilon)

            adv = torch.clamp(img_d + delta, 0, 1)

    return adv.detach().cpu()


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
