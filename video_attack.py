"""
Video Adversarial Attack
========================
Apply adversarial noise to video frames to fool TikTok-like classifiers.

Two modes:
  - Fast: attack one representative frame, apply same noise to all frames (~3-5 min)
  - Quality (warm-start): attack keyframes, interpolate noise between them (~15-40 min)

Requires: opencv-python-headless
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from adversarial_attack import (
    normalise,
    ensemble_mi_di_ti_fgsm,
)


# ── Frame I/O ────────────────────────────────────────────────────────────────

def extract_frames(video_bytes: bytes) -> tuple[list[np.ndarray], float, int, int]:
    """
    Extract all frames from video bytes.
    Returns: (frames_rgb_list, fps, width, height)
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    Path(tmp_path).unlink(missing_ok=True)
    return frames, fps, width, height


def save_video(frames_rgb: list[np.ndarray], fps: float, output_path: str) -> None:
    """Save RGB frames to MP4 file."""
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def frame_to_tensor_224(frame_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB numpy frame to 1x3x224x224 tensor [0,1]."""
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def frame_to_tensor_full(frame_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB numpy frame to 1x3xHxW tensor [0,1]."""
    return transforms.ToTensor()(Image.fromarray(frame_rgb)).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert 1x3xHxW tensor [0,1] to RGB numpy array."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return (img_np * 255).astype(np.uint8)


# ── Fast mode ────────────────────────────────────────────────────────────────

def attack_video_fast(
    ensemble: list[torch.nn.Module],
    video_bytes: bytes,
    target_class: int,
    epsilon: float = 0.06,
    steps: int = 120,
    progress_callback=None,
) -> bytes:
    """
    Fast video attack: compute noise on one representative frame,
    apply the same noise (upscaled) to all frames.

    progress_callback(phase, frame_idx, total_frames, step, total_steps)
    """
    # Extract frames
    if progress_callback:
        progress_callback("extracting", 0, 0, 0, 0)
    frames, fps, orig_w, orig_h = extract_frames(video_bytes)
    total_frames = len(frames)

    # Pick middle frame as representative
    mid_idx = total_frames // 2
    mid_224 = frame_to_tensor_224(frames[mid_idx])

    # Attack the representative frame
    def on_step(step, total):
        if progress_callback:
            progress_callback("attacking", 0, 1, step + 1, total)

    adv_224 = ensemble_mi_di_ti_fgsm(
        ensemble, mid_224, target_class,
        epsilon=epsilon, steps=steps,
        progress_callback=on_step,
    )

    # Compute noise at 224x224, upscale to full resolution
    noise_224 = adv_224 - mid_224
    noise_full = F.interpolate(
        noise_224, size=(orig_h, orig_w),
        mode="bilinear", align_corners=False,
    )

    # Apply same noise to all frames
    if progress_callback:
        progress_callback("applying", 0, total_frames, 0, 0)

    output_frames = []
    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback("applying", i + 1, total_frames, 0, 0)
        full_tensor = frame_to_tensor_full(frame)
        adv_full = torch.clamp(full_tensor + noise_full, 0, 1)
        output_frames.append(tensor_to_frame(adv_full))

    # Save video
    if progress_callback:
        progress_callback("saving", 0, 0, 0, 0)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    save_video(output_frames, fps, tmp_out.name)

    with open(tmp_out.name, "rb") as f:
        result_bytes = f.read()
    Path(tmp_out.name).unlink(missing_ok=True)

    return result_bytes


# ── Quality mode (warm-start) ────────────────────────────────────────────────

def attack_video_warmstart(
    ensemble: list[torch.nn.Module],
    video_bytes: bytes,
    target_class: int,
    epsilon: float = 0.06,
    steps_first: int = 120,
    steps_warm: int = 30,
    keyframe_fps: float = 2.0,
    progress_callback=None,
) -> bytes:
    """
    Quality video attack with warm-start:
    1) Attack keyframes (every 1/keyframe_fps seconds)
    2) First keyframe gets full steps, rest use warm-start from previous noise
    3) Interpolate noise between keyframes
    4) Apply upscaled noise to original frames

    progress_callback(phase, frame_idx, total_frames, step, total_steps)
    """
    # Extract frames
    if progress_callback:
        progress_callback("extracting", 0, 0, 0, 0)
    frames, fps, orig_w, orig_h = extract_frames(video_bytes)
    total_frames = len(frames)

    # Determine keyframe indices
    keyframe_interval = max(1, int(round(fps / keyframe_fps)))
    keyframe_indices = list(range(0, total_frames, keyframe_interval))
    num_keyframes = len(keyframe_indices)
    total_attack_steps = steps_first + (num_keyframes - 1) * steps_warm

    # Attack keyframes
    noise_map: dict[int, torch.Tensor] = {}
    prev_delta = None
    global_step = 0

    for ki, kf_idx in enumerate(keyframe_indices):
        frame_224 = frame_to_tensor_224(frames[kf_idx])

        is_first = ki == 0
        n_steps = steps_first if is_first else steps_warm

        def on_step(step, total, _ki=ki, _kf_idx=kf_idx):
            nonlocal global_step
            if progress_callback:
                current = (global_step + step + 1)
                progress_callback(
                    "attacking", _ki + 1, num_keyframes,
                    current, total_attack_steps,
                )

        adv_224 = ensemble_mi_di_ti_fgsm(
            ensemble, frame_224, target_class,
            epsilon=epsilon, steps=n_steps,
            progress_callback=on_step,
            init_delta=prev_delta,
        )

        delta = adv_224 - frame_224
        noise_map[kf_idx] = delta
        prev_delta = delta
        global_step += n_steps

    # Interpolate noise for all frames (at 224x224)
    if progress_callback:
        progress_callback("interpolating", 0, total_frames, 0, 0)

    all_noise_224: list[torch.Tensor] = []
    for fi in range(total_frames):
        if fi in noise_map:
            all_noise_224.append(noise_map[fi])
        else:
            # Find surrounding keyframes
            prev_kf = max(k for k in keyframe_indices if k <= fi)
            candidates = [k for k in keyframe_indices if k > fi]
            next_kf = candidates[0] if candidates else prev_kf

            if prev_kf == next_kf:
                all_noise_224.append(noise_map[prev_kf])
            else:
                t = (fi - prev_kf) / (next_kf - prev_kf)
                interp = (1 - t) * noise_map[prev_kf] + t * noise_map[next_kf]
                all_noise_224.append(interp)

    # Apply noise to original frames (upscale each noise to full resolution)
    if progress_callback:
        progress_callback("applying", 0, total_frames, 0, 0)

    output_frames = []
    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback("applying", i + 1, total_frames, 0, 0)

        noise_full = F.interpolate(
            all_noise_224[i], size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )
        full_tensor = frame_to_tensor_full(frame)
        adv_full = torch.clamp(full_tensor + noise_full, 0, 1)
        output_frames.append(tensor_to_frame(adv_full))

    # Save video
    if progress_callback:
        progress_callback("saving", 0, 0, 0, 0)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    save_video(output_frames, fps, tmp_out.name)

    with open(tmp_out.name, "rb") as f:
        result_bytes = f.read()
    Path(tmp_out.name).unlink(missing_ok=True)

    return result_bytes
