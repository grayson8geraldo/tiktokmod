"""
Quick demo: download a sample couch image and run the adversarial attack.

Usage:
    python demo.py
"""

import subprocess
import sys
import urllib.request
from pathlib import Path


SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "0/09/Couch_and_chair.JPG/640px-Couch_and_chair.JPG"
)
SAMPLE_PATH = "examples/couch.jpg"


def download_sample_image() -> str:
    Path("examples").mkdir(exist_ok=True)
    if not Path(SAMPLE_PATH).exists():
        print(f"Downloading sample couch image → {SAMPLE_PATH}")
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_PATH)
    else:
        print(f"Sample image already exists: {SAMPLE_PATH}")
    return SAMPLE_PATH


def main() -> None:
    img_path = download_sample_image()

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: FGSM attack  (couch → cat)")
    print("=" * 60)
    subprocess.run([
        sys.executable, "adversarial_attack.py",
        "--image", img_path,
        "--target", "281",       # tabby cat
        "--method", "fgsm",
        "--epsilon", "0.03",
        "--output", "output/fgsm_couch_to_cat.png",
    ], check=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: PGD attack  (couch → cat)")
    print("=" * 60)
    subprocess.run([
        sys.executable, "adversarial_attack.py",
        "--image", img_path,
        "--target", "281",       # tabby cat
        "--method", "pgd",
        "--epsilon", "0.03",
        "--steps", "40",
        "--output", "output/pgd_couch_to_cat.png",
    ], check=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: PGD attack with stronger noise  (ε=0.05)")
    print("=" * 60)
    subprocess.run([
        sys.executable, "adversarial_attack.py",
        "--image", img_path,
        "--target", "281",
        "--method", "pgd",
        "--epsilon", "0.05",
        "--steps", "80",
        "--output", "output/pgd_strong_couch_to_cat.png",
    ], check=True)

    print("\n" + "=" * 60)
    print("  Done! Check the output/ folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
