"""Top-level package for my_niche_utils."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """mayu"""
__email__ = "concorde2591@gmail.com"
__version__ = "0.0.1"

from .src.my_niche_utils.nodes import NODE_CLASS_MAPPINGS
from .src.my_niche_utils.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"
