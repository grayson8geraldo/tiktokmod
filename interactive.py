"""
Interactive Adversarial Attack Tool
====================================
Load your own image, see what the neural network thinks it is,
search for a target class, and transform the image with adversarial noise.

Usage:
    python interactive.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

from adversarial_attack import (
    load_imagenet_labels,
    load_image,
    predict,
    fgsm_targeted,
    pgd_targeted,
    visualise,
    generate_test_image,
)
from torchvision import models


# ── Popular categories for quick selection ────────────────────────────────────
POPULAR_TARGETS = [
    (281, "tabby cat"),
    (283, "Persian cat"),
    (285, "Egyptian cat"),
    (207, "golden retriever"),
    (235, "German shepherd"),
    (1,   "goldfish"),
    (130, "flamingo"),
    (291, "lion"),
    (340, "zebra"),
    (386, "African elephant"),
    (949, "strawberry"),
    (954, "banana"),
    (820, "couch"),
    (417, "balloon"),
    (920, "traffic light"),
    (468, "cab / taxi"),
    (895, "warplane"),
    (812, "space shuttle"),
    (999, "toilet paper"),
]


def print_header():
    print()
    print("=" * 60)
    print("  Interactive Adversarial Attack Tool")
    print("  Turn any object into anything else!")
    print("=" * 60)
    print()


def ask_image_path() -> str | None:
    """Ask user for image path; return None for synthetic image."""
    print("STEP 1: Choose an image")
    print("-" * 40)
    print("  Enter the path to your image (jpg, png, etc.)")
    print("  Or press Enter for a synthetic test image.")
    print()
    path = input("  Image path: ").strip().strip("'\"")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"\n  [!] File not found: {path}")
        print("      Trying anyway in case it's a relative path...")
    return path


def show_top_predictions(model, img, labels, top_k=5):
    """Show the model's top-k predictions for the image."""
    with torch.no_grad():
        from adversarial_attack import normalise
        logits = model(normalise(img))
    probs = torch.nn.functional.softmax(logits, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)

    print(f"\n  Model's top-{top_k} predictions:")
    print("  " + "-" * 38)
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item() * 100
        marker = " <-- current" if i == 0 else ""
        print(f"  {i + 1}. {labels[idx]:30s} {prob:5.1f}%{marker}")
    print()


def ask_target_class(labels: list[str]) -> int:
    """Let user choose what to turn the image into."""
    print("STEP 2: Choose what to turn it into")
    print("-" * 40)
    print()
    print("  Popular targets:")
    for i, (idx, name) in enumerate(POPULAR_TARGETS):
        col_end = "\n" if (i + 1) % 2 == 0 else ""
        print(f"    {i + 1:2d}. {name:20s} (#{idx})", end=col_end)
    if len(POPULAR_TARGETS) % 2 != 0:
        print()
    print()
    print("  Options:")
    print("    - Enter a number (1-{}) for a popular target".format(
        len(POPULAR_TARGETS)))
    print("    - Enter a class index (0-999) directly with '#' prefix, e.g. #207")
    print("    - Type a search term to find a class, e.g. 'shark'")
    print()

    while True:
        choice = input("  Your choice: ").strip()
        if not choice:
            print("  Using default: tabby cat (#281)")
            return 281

        # Option 1: number from popular list
        if choice.isdigit() and 1 <= int(choice) <= len(POPULAR_TARGETS):
            idx, name = POPULAR_TARGETS[int(choice) - 1]
            print(f"  -> Selected: {name} (#{idx})")
            return idx

        # Option 2: direct class index with # prefix
        if choice.startswith("#") and choice[1:].isdigit():
            idx = int(choice[1:])
            if 0 <= idx < len(labels):
                print(f"  -> Selected: {labels[idx]} (#{idx})")
                return idx
            print(f"  [!] Index {idx} out of range (0-999). Try again.")
            continue

        # Option 3: search by keyword
        query = choice.lower()
        matches = [
            (i, label) for i, label in enumerate(labels)
            if query in label.lower()
        ]

        if not matches:
            print(f"  [!] No classes found matching '{choice}'. Try another term.")
            continue

        if len(matches) == 1:
            idx, name = matches[0]
            print(f"  -> Found: {name} (#{idx})")
            return idx

        print(f"\n  Found {len(matches)} matching classes:")
        for j, (idx, name) in enumerate(matches[:20]):
            print(f"    {j + 1:2d}. {name:30s} (#{idx})")
        if len(matches) > 20:
            print(f"    ... and {len(matches) - 20} more. Try a more specific term.")
        print()

        sub = input("  Pick a number from the list (or press Enter to search again): ").strip()
        if sub.isdigit() and 1 <= int(sub) <= min(len(matches), 20):
            idx, name = matches[int(sub) - 1]
            print(f"  -> Selected: {name} (#{idx})")
            return idx


def ask_attack_settings() -> dict:
    """Let user configure attack parameters."""
    print("\nSTEP 3: Attack settings")
    print("-" * 40)
    print()
    print("  Method:")
    print("    1. PGD  — iterative, strong (recommended)")
    print("    2. FGSM — single-step, fast but weaker")
    print()

    method_choice = input("  Method [1]: ").strip()
    method = "fgsm" if method_choice == "2" else "pgd"

    print()
    print("  Noise strength (epsilon):")
    print("    - 0.01 = very subtle (hard to succeed)")
    print("    - 0.03 = balanced (default)")
    print("    - 0.05 = stronger (more visible noise)")
    print("    - 0.10 = aggressive (noticeable artifacts)")
    print()

    eps_input = input("  Epsilon [0.03]: ").strip()
    try:
        epsilon = float(eps_input) if eps_input else 0.03
    except ValueError:
        epsilon = 0.03

    steps = 40
    if method == "pgd":
        steps_input = input("  PGD steps [40]: ").strip()
        try:
            steps = int(steps_input) if steps_input else 40
        except ValueError:
            steps = 40

    return {"method": method, "epsilon": epsilon, "steps": steps}


def ask_output_path() -> str:
    """Ask where to save the result."""
    print()
    default = "output/result.png"
    path = input(f"  Save result to [{default}]: ").strip()
    return path if path else default


def run_interactive():
    print_header()

    # Load model and labels
    print("Loading model and labels (this may take a moment on first run)...")
    labels = load_imagenet_labels()
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    print("Model ready!\n")

    while True:
        # Step 1: Image
        image_path = ask_image_path()
        if image_path:
            try:
                img = load_image(image_path)
                print(f"  -> Loaded: {image_path}")
            except Exception as e:
                print(f"  [!] Error loading image: {e}")
                print("      Using synthetic test image instead.")
                img = generate_test_image()
        else:
            print("  -> Using synthetic test image")
            img = generate_test_image()

        # Show what the model currently sees
        print()
        show_top_predictions(model, img, labels)

        # Step 2: Target
        target_class = ask_target_class(labels)

        # Step 3: Settings
        settings = ask_attack_settings()

        # Step 4: Output path
        output_path = ask_output_path()

        # Run the attack
        print()
        print("=" * 60)
        method_name = settings["method"].upper()
        print(f"  Running {method_name} attack → target: "
              f"{labels[target_class]} (#{target_class})")
        print(f"  epsilon={settings['epsilon']}, "
              f"steps={settings['steps']}")
        print("=" * 60)
        print()

        orig_idx, orig_label, orig_conf = predict(model, img, labels)
        print(f"  Before: {orig_label} ({orig_conf:.1f}%)")

        if settings["method"] == "fgsm":
            adv_img = fgsm_targeted(model, img, target_class,
                                     settings["epsilon"])
        else:
            adv_img = pgd_targeted(model, img, target_class,
                                    settings["epsilon"],
                                    steps=settings["steps"])

        adv_idx, adv_label, adv_conf = predict(model, adv_img, labels)
        print(f"  After:  {adv_label} ({adv_conf:.1f}%)")

        if adv_idx == target_class:
            print(f"\n  SUCCESS! The model now sees '{adv_label}' "
                  f"with {adv_conf:.1f}% confidence!")
        else:
            print(f"\n  Target class not reached (got '{adv_label}' instead).")
            print("  Tip: try increasing epsilon or steps.")

        # Show top-5 after attack
        show_top_predictions(model, adv_img, labels)

        # Save
        visualise(img, adv_img, orig_label, orig_conf,
                  adv_label, adv_conf, save_path=output_path)

        # Save adversarial image separately
        adv_save_path = str(Path(output_path).with_stem(
            Path(output_path).stem + "_adversarial"))
        from torchvision.utils import save_image
        Path(adv_save_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(adv_img.squeeze(0), adv_save_path)
        print(f"[✓] Saved adversarial image → {adv_save_path}")

        # Noise stats
        noise = adv_img - img
        l_inf = noise.abs().max().item()
        print(f"\n  Noise: L∞ = {l_inf:.4f} "
              f"(max pixel change: {l_inf * 255:.1f}/255)")

        # Again?
        print()
        again = input("  Run another experiment? [Y/n]: ").strip().lower()
        if again in ("n", "no", "q", "quit", "exit"):
            print("\n  Bye!\n")
            break
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_interactive()
