# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# TO RUN THIS NOTEBOOK YOU FIRST NEED TO MAKE ALL OF THE FOLDERS THAT NEED TO BE POPULATED

# **Essential image preprocessing:**
# - ensure they are all square
# - image scaling, ensuring they all have the same height and width
# - normalising image inputs (get some before and after examples to include in the report)
# - remove duplicates
#
# **Interesting ideas for further analysis:**
# - dimensionality reduction )eg. making all images greyscale
# - data augmentation to increase dataset size, eg, rotating, scaling etc.
# - blurring to replicate bad user photos
#
# resource: https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258

# +
from pathlib import Path
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms


base_dir = Path("..", "data", "1_original_images")
new_dir = Path("..", "data", "2_preprocessed")


# -


def preprocess_images(current_dir, new_dir):
    examples = list(current_dir.glob("*"))

    for i, image in enumerate(examples):
        im = Image.open(str(image))

        padded = PIL.ImageOps.pad(
            im, size=(224, 224), color="white", centering=(0.5, 0.5)
        )

        padded.save(new_dir.joinpath(f"{i}.jpg"))


# # Example Folder

example_dir = Path("..", "data", "report_examples", "report_examples")
# new_example_dir = Path('..','data', 'report_examples', 'preprocessed_report_examples')
# preprocess_images(example_dir, new_example_dir)

# +
examples = list(example_dir.glob("*"))

image = Image.open(str(examples[0]))
plt.imshow(image)

# +
image = transform(image)
transform_back = transforms.ToPILImage()
img = transform_back(image)

plt.imshow(img)
# -

# # Christmas

christmas = base_dir.joinpath("christmas")
new_christmas = new_dir.joinpath("christmas")
preprocess_images(christmas, new_christmas)

# # Normal

normal = base_dir.joinpath("normal")
new_normal = new_dir.joinpath("normal")
preprocess_images(normal, new_normal)

# # Black and White version

base_dir = Path("..", "data", "2_preprocessed")
new_dir = Path("..", "data", "4_black_and_white")


def greyscale_images(current_dir, new_dir):
    examples = list(current_dir.glob("*"))

    for i, image in enumerate(examples):
        im = Image.open(str(image))
        bw_image = im.convert("L")
        bw_image.save(new_dir.joinpath(f"{i}.jpg"))


christmas = base_dir.joinpath("christmas")
new_christmas = new_dir.joinpath("christmas")
greyscale_images(christmas, new_christmas)

normal = base_dir.joinpath("normal")
new_normal = new_dir.joinpath("normal")
greyscale_images(normal, new_normal)
