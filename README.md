# Adversarial Noise Experiment

Demonstrates how imperceptible mathematical noise can fool a neural network
into misclassifying an image (e.g. a **couch** becomes a **cat**).

## How it works

A pre-trained ResNet-50 model correctly classifies an input image.
Then we compute a small perturbation (adversarial noise) using gradient-based
methods so that the model confidently predicts a completely different class.

Two attack methods are implemented:

| Method | Description |
|--------|-------------|
| **FGSM** | Fast Gradient Sign Method — single-step, fast but weaker |
| **PGD**  | Projected Gradient Descent — iterative, stronger attack |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Interactive mode (recommended)

Choose your own image and target class step-by-step:

```bash
python interactive.py
```

The interactive tool will:
1. Ask you for an image (or use a test image)
2. Show the model's current top-5 predictions
3. Let you search and pick a target class (e.g. type "shark" to find shark classes)
4. Configure attack strength
5. Run the attack, show results, and save the visualisation

### Quick demo (downloads a sample couch image automatically)

```bash
python demo.py
```

### Manual run

```bash
# Classify couch.jpg, then attack it to become "tabby cat" (class 281)
python adversarial_attack.py --image examples/couch.jpg --target 281 --method pgd

# FGSM attack with custom epsilon
python adversarial_attack.py --image examples/couch.jpg --target 281 --method fgsm --epsilon 0.05

# Without an image (uses synthetic test image)
python adversarial_attack.py --target 281
```

### Common ImageNet class indices

| Index | Label |
|-------|-------|
| 281   | tabby cat |
| 283   | Persian cat |
| 285   | Egyptian cat |
| 207   | golden retriever |
| 820   | couch / sofa |
| 999   | toilet tissue |

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | *(synthetic)* | Path to input image |
| `--target` | `281` | Target ImageNet class index |
| `--method` | `pgd` | Attack method: `fgsm` or `pgd` |
| `--epsilon` | `0.03` | Max perturbation (L-inf norm) |
| `--steps` | `40` | PGD iteration count |
| `--output` | `output/result.png` | Where to save the visualisation |

## Output

The program generates a 3-panel image:

```
[ Original image ]  [ Amplified noise ]  [ Adversarial image ]
  couch (92.3%)                            tabby cat (87.1%)
```

The noise panel is amplified 10x for visibility — the actual perturbation
is imperceptible to the human eye.
