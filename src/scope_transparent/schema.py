"""Configuration schema for Transparent pipeline."""

from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class TransparentConfig(BasePipelineConfig):
    pipeline_id = "transparent"
    pipeline_name = "Transparent"
    pipeline_description = (
        "Applies a mask to make parts of a video transparent. "
        "Accepts video and mask inputs. Without a mask, acts as passthrough."
    )
    artifacts = []
    inputs = ["video", "mask"]
    outputs = ["video"]
    supports_prompts = False
    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mask threshold: pixels above this become transparent",
        json_schema_extra=ui_field_config(order=1, label="Threshold"),
    )
    invert_mask: bool = Field(
        default=False,
        description="Invert the mask (swap transparent and opaque regions)",
        json_schema_extra=ui_field_config(order=2, label="Invert Mask"),
    )
    background: Literal["black", "checkerboard"] = Field(
        default="checkerboard",
        description="Background to show through transparent regions",
        json_schema_extra=ui_field_config(order=3, label="Background"),
    )
