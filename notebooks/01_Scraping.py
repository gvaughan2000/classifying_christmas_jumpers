# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:light
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
from jmd_imagescraper.core import *  # dont't worry, it's designed to work with import *
from pathlib import Path


root = Path().cwd() / "images"

folder = "not square"
search_terms = "Christmas sweater"

params = {
    "max_results": 500,
    "img_type": ImgType.Transparent,
    "img_layout": ImgLayout.All,
    "img_color": ImgColor.All,
    "uuid_names": True,
}

# duckduckgo_search(root, folder, search_terms, **params)
# -
