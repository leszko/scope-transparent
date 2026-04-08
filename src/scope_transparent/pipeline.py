"""Transparent mask pipeline."""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import TransparentConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _make_checkerboard(h: int, w: int, device: torch.device, square: int = 16):
    """Create a checkerboard pattern [H, W, 3] float [0, 1]."""
    ys = torch.arange(h, device=device) // square
    xs = torch.arange(w, device=device) // square
    grid = (ys[:, None] + xs[None, :]) % 2  # [H, W] binary
    light, dark = 0.8, 0.6
    pattern = torch.where(grid == 0, light, dark)  # [H, W]
    return pattern.unsqueeze(-1).expand(h, w, 3)


class TransparentPipeline(Pipeline):
    """Applies a mask to make parts of a video transparent.

    Inputs:
        video: RGB video frames
        mask: Mask frames (RGB or grayscale) - bright = transparent
    Output:
        video: Frames with masked areas replaced by background
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TransparentConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None")

        mask_input = kwargs.get("mask")

        # No mask -> passthrough
        if mask_input is None:
            video = normalize_frame_sizes(video, device=self.device)
            frames = torch.cat(video, dim=0)  # [T, H, W, C]
            return {"video": (frames.float() / 255.0).clamp(0, 1)}

        # Normalize both inputs to same size
        video = normalize_frame_sizes(video, device=self.device)
        mask_input = normalize_frame_sizes(
            mask_input,
            target_height=video[0].shape[1],
            target_width=video[0].shape[2],
            device=self.device,
        )

        threshold = kwargs.get("threshold", 0.5)
        invert_mask = kwargs.get("invert_mask", False)
        background = kwargs.get("background", "checkerboard")

        result_frames = []
        for i, frame in enumerate(video):
            # frame is [1, H, W, C] uint8
            mask_frame = mask_input[min(i, len(mask_input) - 1)]

            frame_f = frame.squeeze(0).float() / 255.0  # [H, W, C]
            mask_f = mask_frame.squeeze(0).float() / 255.0  # [H, W, C]

            # Convert mask to grayscale
            if mask_f.shape[-1] == 3:
                gray = (
                    0.299 * mask_f[..., 0]
                    + 0.587 * mask_f[..., 1]
                    + 0.114 * mask_f[..., 2]
                )
            else:
                gray = mask_f[..., 0]

            # Binary threshold
            binary_mask = (gray > threshold).float()
            if invert_mask:
                binary_mask = 1.0 - binary_mask

            # binary_mask: 1 = transparent, 0 = keep video
            h, w = frame_f.shape[:2]
            mask_3ch = binary_mask.unsqueeze(-1)  # [H, W, 1]

            if background == "checkerboard":
                bg = _make_checkerboard(h, w, self.device)
            else:
                bg = torch.zeros(h, w, 3, device=self.device)

            out = frame_f * (1.0 - mask_3ch) + bg * mask_3ch
            result_frames.append(out)

        video_out = torch.stack(result_frames, dim=0).cpu()  # [T, H, W, C]
        return {"video": video_out.clamp(0, 1)}
