from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image, ImageFilter, ImageOps


MODEL_NAME = "depth-anything/DA3-SMALL"
IMAGE_SIZE = (512, 512)
PROCESS_RESOLUTION = 504
MAX_BLUR_SIGMA = 15
INPUT_PATH = Path(__file__).resolve().parent / "my_image2.jpeg"
DEPTH_MAP_PATH = Path(__file__).resolve().parent / "da3_depth_map.png"
COMPARISON_PATH = Path(__file__).resolve().parent / "da3_depth_comparison.png"
DEPTH_VALUES_PATH = Path(__file__).resolve().parent / "da3_depth_0_15.npy"
LENS_BLUR_PATH = Path(__file__).resolve().parent / "da3_lens_blur_comparison.png"


# Use the same orientation-corrected, centered 512x512 crop as the segmentation
# workflow so the input shown in the comparison is aligned to the same image.
with Image.open(INPUT_PATH) as source_image:
    image = ImageOps.exif_transpose(source_image).convert("RGB")
image = ImageOps.fit(image, IMAGE_SIZE)

# DA3's processing resolution is a model input size, which it adjusts to fit its
# patch grid. Use a patch-aligned resolution close to 512, then resize the
# predicted depth visualization back to the exact input dimensions below.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained(MODEL_NAME).to(device=device)
model.eval()

with torch.inference_mode():
    prediction = model.inference(
        [image],
        process_res=PROCESS_RESOLUTION,
        process_res_method="upper_bound_resize",
    )

# DA3 returns relative depth values as [images, height, width]. Normalize the
# predicted depth directly: lower values are displayed darker (nearer), while
# higher values are brighter (farther).
depth = np.asarray(prediction.depth[0], dtype=np.float32)
finite_pixels = np.isfinite(depth)
if not finite_pixels.any():
    raise ValueError("The depth model returned no finite depth values.")

nearest_value = float(depth[finite_pixels].min())
farthest_value = float(depth[finite_pixels].max())
if farthest_value <= nearest_value:
    raise ValueError("The depth model returned a constant depth map.")

depth_0_15 = np.zeros(depth.shape, dtype=np.float32)
depth_0_15[finite_pixels] = (
    (depth[finite_pixels] - nearest_value)
    / (farthest_value - nearest_value)
    * MAX_BLUR_SIGMA
)
# Resize the floating-point depth values to match the 512x512 image before
# choosing per-pixel blur strengths, preserving fractional depth values.
depth_0_15 = np.asarray(
    Image.fromarray(depth_0_15).resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    ),
    dtype=np.float32,
)
depth_0_15_min = float(depth_0_15.min())
depth_0_15_max = float(depth_0_15.max())
if depth_0_15_max <= depth_0_15_min:
    raise ValueError("The resized depth map has no range to normalize.")
depth_0_15 = (
    (depth_0_15 - depth_0_15_min)
    / (depth_0_15_max - depth_0_15_min)
    * MAX_BLUR_SIGMA
)
np.save(DEPTH_VALUES_PATH, depth_0_15)

# Save a grayscale preview of the 0-to-15 depth values; brighter pixels are
# farther from the camera, with the full floating-point values saved separately.
depth_pixels = np.round(depth_0_15 / MAX_BLUR_SIGMA * 255).astype(np.uint8)
depth_map = Image.fromarray(depth_pixels)
depth_map.save(DEPTH_MAP_PATH)

# Precompute one image for each integer sigma from 0 through 15. For each
# pixel, linearly blend the two neighboring blur levels according to its
# fractional depth, avoiding hard transitions between discrete blur layers.
blurred_layers = np.stack(
    [
        np.asarray(
            image if sigma == 0 else image.filter(ImageFilter.GaussianBlur(radius=sigma)),
            dtype=np.float32,
        )
        for sigma in range(MAX_BLUR_SIGMA + 1)
    ],
    axis=0,
)
lower_sigma = np.floor(depth_0_15).astype(np.intp)
upper_sigma = np.minimum(lower_sigma + 1, MAX_BLUR_SIGMA)
blend = (depth_0_15 - lower_sigma)[..., np.newaxis]
row_indices, column_indices = np.indices(depth_0_15.shape)
lower_blur = blurred_layers[lower_sigma, row_indices, column_indices]
upper_blur = blurred_layers[upper_sigma, row_indices, column_indices]
blurred_pixels = np.rint(lower_blur * (1.0 - blend) + upper_blur * blend)
depth_blurred_image = Image.fromarray(np.clip(blurred_pixels, 0, 255).astype(np.uint8))

# Compare the original input with the variable-depth lens-blur result.
lens_blur_comparison = Image.new("RGB", (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1]))
lens_blur_comparison.paste(image, (0, 0))
lens_blur_comparison.paste(depth_blurred_image, (IMAGE_SIZE[0], 0))
lens_blur_comparison.save(LENS_BLUR_PATH)

# Keep the earlier input/depth-map comparison available as a separate output.
comparison = Image.new("RGB", (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1]))
comparison.paste(image, (0, 0))
comparison.paste(depth_map.convert("RGB"), (IMAGE_SIZE[0], 0))
comparison.save(COMPARISON_PATH)

print(f"Depth map saved to: {DEPTH_MAP_PATH}")
print(f"0-to-15 depth values saved to: {DEPTH_VALUES_PATH}")
print(f"Side-by-side comparison saved to: {COMPARISON_PATH}")
print(f"Depth-based lens blur comparison saved to: {LENS_BLUR_PATH}")
