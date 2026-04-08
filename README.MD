# Transparent Plugin for Daydream Scope

![Transparent plugin in Scope](https://raw.githubusercontent.com/leszko/scope-transparent/main/screenshot.png)

Applies a transparency mask to video frames:

- **video**: the source video stream.
- **mask**: a grayscale mask that determines which areas become transparent.

If no mask is connected, the video passes through unchanged. When a mask is provided, pixels above the configured threshold become transparent, revealing the selected background.

## Configuration

- **threshold** (0.0–1.0): mask intensity cutoff for transparency (default: 0.5).
- **invert_mask** (true/false): swap transparent and opaque regions.
- **background** ("checkerboard" or "black"): what to show behind transparent areas.
