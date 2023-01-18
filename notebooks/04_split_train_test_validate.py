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

# +
import python_splitter
from pathlib import Path

data_dir = Path("..", "data", "4_black_and_white")

# +
# Note: This will save the folders into the current directory (eg. in the notebooks folder)
# -

python_splitter.split_from_folder(data_dir, train=0.5, test=0.3, val=0.2)


