from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


MODEL_NAME = "shi-labs/oneformer_cityscapes_swin_large"
# Pillow sizes are specified as (width, height); the requested output is square.
IMAGE_SIZE = (512, 512)
BLUR_SIGMA = 15
# Resolve paths from this script so it also works when launched from another folder.
IMAGE_PATH = Path(__file__).resolve().parent / "my_image2.jpeg"
MASK_PATH = Path(__file__).resolve().parent / "person_mask2.png"
COMPARISON_PATH = Path(__file__).resolve().parent / "background_blur_comparison.png"


# Respect the camera's EXIF orientation, discard alpha if present, and prepare
# the fixed-size model input by scaling to fill the square and center-cropping.
with Image.open(IMAGE_PATH) as source_image:
    image = ImageOps.exif_transpose(source_image).convert("RGB")
image = ImageOps.fit(image, IMAGE_SIZE)

# The OneFormer processor supplies the checkpoint's image normalization and
# task-token inputs. This checkpoint is prompted explicitly for semantic output.
processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME)

# Look up the class ID from the checkpoint rather than relying on a label-order
# assumption; Cityscapes assigns the "person" label its own semantic class.
try:
    person_class_id = next(
        class_id
        for class_id, label in model.config.id2label.items()
        if label.casefold() == "person"
    )
except StopIteration as exc:
    raise ValueError(f"The model {MODEL_NAME!r} has no 'person' class.") from exc

# Use a GPU when available, otherwise run on CPU. Evaluation mode disables
# training-only behavior, and inference_mode avoids storing gradients.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

inputs = processor(
    images=image,
    task_inputs=["semantic"],
    return_tensors="pt",
).to(device)

# Run the forward pass without building an autograd graph, since this script
# only needs predictions.
with torch.inference_mode():
    outputs = model(**inputs)

# OneFormer predicts class scores and masks for multiple queries. Its semantic
# postprocessor combines those scores into the highest-scoring class per pixel
# and resizes the resulting class map to (height, width) for the 512x512 image.
predicted_classes = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=[(IMAGE_SIZE[1], IMAGE_SIZE[0])],
)[0]

# Keep only the person class and encode it as an 8-bit grayscale mask:
# person pixels are 255 (white), and every other class is 0 (black).
person_mask = (predicted_classes == person_class_id).to(torch.uint8).cpu().numpy() * 255

# Save the binary mask at the same dimensions as the preprocessed image, ready
# for later compositing or visualization.
mask_image = Image.fromarray(person_mask)
mask_image.save(MASK_PATH)

# Pillow's GaussianBlur radius is the Gaussian standard deviation (sigma).
# Blur the full image, then use the person mask as an alpha selection: white
# mask pixels keep the sharp original and black pixels reveal the blurred copy.
blurred_background = image.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
segmented_image = Image.composite(image, blurred_background, mask_image)

# Make a comparison canvas with the cropped input on the left and the
# background-blurred result on the right, both at the same 512x512 size.
comparison = Image.new("RGB", (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1]))
comparison.paste(image, (0, 0))
comparison.paste(segmented_image, (IMAGE_SIZE[0], 0))
comparison.save(COMPARISON_PATH)
