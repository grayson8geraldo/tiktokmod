"""
Video Adversarial Attack
========================
Apply adversarial noise to video frames to fool TikTok-like classifiers.

Two modes:
  - Fast: attack one representative frame, apply same noise to all frames (~3-5 min)
  - Quality (warm-start): attack keyframes, interpolate noise between them (~15-40 min)

Requires: opencv-python-headless
Audio preservation requires: ffmpeg (system binary)
"""

import io
import shutil
import subprocess
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


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio track from video to a separate file. Returns True if audio exists."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0 and Path(audio_path).stat().st_size > 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """Mux video (no audio) + audio into a single file."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ],
            capture_output=True, timeout=120,
        )
        return result.returncode == 0 and Path(output_path).stat().st_size > 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ── Frame I/O ────────────────────────────────────────────────────────────────

def extract_frames_from_path(video_path: str) -> tuple[list[np.ndarray], float, int, int]:
    """
    Extract all frames from a video file on disk.
    Returns: (frames_rgb_list, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
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
    return frames, fps, width, height


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

    result = extract_frames_from_path(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return result


def save_video(frames_rgb: list[np.ndarray], fps: float, output_path: str) -> None:
    """Save RGB frames to MP4 file (H.264 for browser compatibility)."""
    h, w = frames_rgb[0].shape[:2]
    # Try H.264 first (browser-compatible), fallback to mp4v
    for codec in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
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


# ── Temporal flickering ───────────────────────────────────────────────────────

def _apply_temporal_flicker(
    all_noise: list[torch.Tensor],
    num_frames: int,
    intensity: float = 0.3,
    temporal_smoothness: float = 0.7,
    epsilon: float = 0.06,
) -> list[torch.Tensor]:
    """
    Add temporally correlated random variation to per-frame noise.

    Uses a random walk with momentum so the flicker looks natural (camera-sensor
    noise) rather than like random static, which would be perceptible.

    Args:
        all_noise: list of 1x3xHxW noise tensors (one per frame)
        num_frames: number of output frames (must == len(all_noise))
        intensity: fraction of epsilon used for flicker amplitude (0 = off)
        temporal_smoothness: momentum factor (0 = fully random, 1 = static)
        epsilon: global perturbation budget

    Returns:
        list of 1x3xHxW flickered noise tensors, re-projected into [-eps, eps]
    """
    if intensity <= 0 or num_frames <= 1:
        return all_noise

    flicker_eps = epsilon * intensity
    result = []

    # Random walk state — starts at zero, drifts with momentum
    prev_variation = torch.zeros_like(all_noise[0])

    for t in range(num_frames):
        base = all_noise[t]

        # New random direction (normalised to unit mean-abs)
        new_dir = torch.randn_like(base)
        new_dir = new_dir / (new_dir.abs().mean() + 1e-8)

        # Blend with previous frame's variation (temporal smoothness)
        variation = temporal_smoothness * prev_variation + (1 - temporal_smoothness) * new_dir
        variation = variation * flicker_eps

        # Combine with base noise and re-project into epsilon ball
        frame_noise = torch.clamp(base + variation, -epsilon, epsilon)

        result.append(frame_noise)
        prev_variation = variation / (flicker_eps + 1e-8)

    return result


# ── Fast mode ────────────────────────────────────────────────────────────────

def attack_video_fast(
    ensemble: list[torch.nn.Module],
    video_bytes: bytes,
    target_class: int,
    epsilon: float = 0.06,
    steps: int = 120,
    progress_callback=None,
    compression_robust: bool = True,
    face_model=None,
    face_detector=None,
    text_detector=None,
    flicker_intensity: float = 0.0,
) -> bytes:
    """
    Fast video attack: compute noise on one representative frame,
    apply the same noise (upscaled) to all frames.

    face_model: InceptionResnetV1 for face embedding attack (celebrity mode)
    face_detector: callable(tensor_224) -> (x1,y1,x2,y2) or None
    text_detector: callable(frame_rgb_numpy) -> list[(x1,y1,x2,y2)] or None
    flicker_intensity: 0.0-1.0, temporal noise variation between frames

    progress_callback(phase, frame_idx, total_frames, step, total_steps)
    """
    # Save input to temp file (reused for frame extraction + audio extraction)
    tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_input.write(video_bytes)
    tmp_input.flush()
    tmp_input.close()
    input_path = tmp_input.name

    try:
        # Extract audio track (if ffmpeg available)
        audio_path = input_path + ".audio.aac"
        has_audio = _has_ffmpeg() and _extract_audio(input_path, audio_path)

        # Extract frames
        if progress_callback:
            progress_callback("extracting", 0, 0, 0, 0)
        frames, fps, orig_w, orig_h = extract_frames_from_path(input_path)
        total_frames = len(frames)

        # Pick middle frame as representative
        mid_idx = total_frames // 2
        mid_224 = frame_to_tensor_224(frames[mid_idx])

        # Detect face for celebrity mode
        face_box = None
        if face_model is not None and face_detector is not None:
            face_box = face_detector(mid_224)

        # Detect text regions for OCR protection (full-res → 224 coords)
        text_boxes = None
        if text_detector is not None:
            text_boxes = text_detector(frames[mid_idx])

        # Attack the representative frame
        def on_step(step, total):
            if progress_callback:
                progress_callback("attacking", 0, 1, step + 1, total)

        adv_224 = ensemble_mi_di_ti_fgsm(
            ensemble, mid_224, target_class,
            epsilon=epsilon, steps=steps,
            progress_callback=on_step,
            compression_robust=compression_robust,
            face_model=face_model if face_box else None,
            face_box=face_box,
            text_boxes=text_boxes,
        )

        # Compute noise at 224x224
        noise_224 = adv_224 - mid_224

        # Apply temporal flickering (per-frame variation at 224 level)
        all_noise_224 = [noise_224] * total_frames
        if flicker_intensity > 0:
            all_noise_224 = _apply_temporal_flicker(
                all_noise_224, total_frames,
                intensity=flicker_intensity, epsilon=epsilon,
            )

        # Apply noise to all frames (upscale each to full resolution)
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

        # Save video (frames only)
        if progress_callback:
            progress_callback("saving", 0, 0, 0, 0)

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.close()
        save_video(output_frames, fps, tmp_video.name)

        # Mux audio back in if available
        if has_audio:
            tmp_final = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_final.close()
            if _mux_audio(tmp_video.name, audio_path, tmp_final.name):
                # Use the muxed version
                with open(tmp_final.name, "rb") as f:
                    result_bytes = f.read()
                Path(tmp_final.name).unlink(missing_ok=True)
            else:
                # Fallback to video-only
                with open(tmp_video.name, "rb") as f:
                    result_bytes = f.read()
                Path(tmp_final.name).unlink(missing_ok=True)
        else:
            with open(tmp_video.name, "rb") as f:
                result_bytes = f.read()

        Path(tmp_video.name).unlink(missing_ok=True)
        return result_bytes

    finally:
        # Cleanup temp files
        Path(input_path).unlink(missing_ok=True)
        Path(input_path + ".audio.aac").unlink(missing_ok=True)


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
    compression_robust: bool = True,
    face_model=None,
    face_detector=None,
    text_detector=None,
    flicker_intensity: float = 0.0,
) -> bytes:
    """
    Quality video attack with warm-start:
    1) Attack keyframes (every 1/keyframe_fps seconds)
    2) First keyframe gets full steps, rest use warm-start from previous noise
    3) Interpolate noise between keyframes
    4) Apply temporal flickering (per-frame variation)
    5) Apply upscaled noise to original frames

    face_model: InceptionResnetV1 for face embedding attack (celebrity mode)
    face_detector: callable(tensor_224) -> (x1,y1,x2,y2) or None
    text_detector: callable(frame_rgb_numpy) -> list[(x1,y1,x2,y2)] or None
    flicker_intensity: 0.0-1.0, temporal noise variation between frames

    progress_callback(phase, frame_idx, total_frames, step, total_steps)
    """
    # Save input to temp file (reused for frame extraction + audio extraction)
    tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_input.write(video_bytes)
    tmp_input.flush()
    tmp_input.close()
    input_path = tmp_input.name

    try:
        # Extract audio track (if ffmpeg available)
        audio_path = input_path + ".audio.aac"
        has_audio = _has_ffmpeg() and _extract_audio(input_path, audio_path)

        # Extract frames
        if progress_callback:
            progress_callback("extracting", 0, 0, 0, 0)
        frames, fps, orig_w, orig_h = extract_frames_from_path(input_path)
        total_frames = len(frames)

        # Determine keyframe indices
        keyframe_interval = max(1, int(round(fps / keyframe_fps)))
        keyframe_indices = list(range(0, total_frames, keyframe_interval))
        num_keyframes = len(keyframe_indices)
        total_attack_steps = steps_first + (num_keyframes - 1) * steps_warm

        # Detect text once (overlays are usually static across frames)
        text_boxes = None
        if text_detector is not None:
            text_boxes = text_detector(frames[0])

        # Attack keyframes
        noise_map: dict[int, torch.Tensor] = {}
        prev_delta = None
        global_step = 0

        for ki, kf_idx in enumerate(keyframe_indices):
            frame_224 = frame_to_tensor_224(frames[kf_idx])

            # Detect face per keyframe (celebrity mode)
            face_box = None
            if face_model is not None and face_detector is not None:
                face_box = face_detector(frame_224)

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
                compression_robust=compression_robust,
                face_model=face_model if face_box else None,
                face_box=face_box,
                text_boxes=text_boxes,
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

        # Apply temporal flickering on top of interpolated noise
        if flicker_intensity > 0:
            all_noise_224 = _apply_temporal_flicker(
                all_noise_224, total_frames,
                intensity=flicker_intensity, epsilon=epsilon,
            )

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

        # Save video (frames only)
        if progress_callback:
            progress_callback("saving", 0, 0, 0, 0)

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.close()
        save_video(output_frames, fps, tmp_video.name)

        # Mux audio back in if available
        if has_audio:
            tmp_final = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_final.close()
            if _mux_audio(tmp_video.name, audio_path, tmp_final.name):
                with open(tmp_final.name, "rb") as f:
                    result_bytes = f.read()
                Path(tmp_final.name).unlink(missing_ok=True)
            else:
                with open(tmp_video.name, "rb") as f:
                    result_bytes = f.read()
                Path(tmp_final.name).unlink(missing_ok=True)
        else:
            with open(tmp_video.name, "rb") as f:
                result_bytes = f.read()

        Path(tmp_video.name).unlink(missing_ok=True)
        return result_bytes

    finally:
        # Cleanup temp files
        Path(input_path).unlink(missing_ok=True)
        Path(input_path + ".audio.aac").unlink(missing_ok=True)
