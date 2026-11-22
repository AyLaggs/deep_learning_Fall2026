import os
import random

from PIL import Image


IMAGE_SIZE = 64
NUM_IMAGES = 1000
OUTPUT_DIR = "squares"


def generate_dataset(
    output_dir: str = OUTPUT_DIR,
    num_images: int = NUM_IMAGES,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        size = random.randint(10, 35)
        color = tuple(random.randint(50, 255) for _ in range(3))
        square = Image.new("RGBA", (size, size), color + (255,))

        rotated = square.rotate(random.uniform(0, 180), expand=True)
        x = random.randint(0, IMAGE_SIZE - rotated.width)
        y = random.randint(0, IMAGE_SIZE - rotated.height)
        image.paste(rotated, (x, y), rotated)
        image.save(os.path.join(output_dir, f"{i}.png"))


if __name__ == "__main__":
    generate_dataset()
