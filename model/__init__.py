"""
Bilingual Image Captioning Model Package

This package contains models and utilities for generating bilingual 
(English & Vietnamese) image captions using state-of-the-art vision-language models.
"""

from .captioning_models import BLIP2Captioner, BLIPCaptioner
from .translator import VietnameseTranslator
from .utils import load_image, preprocess_image

__version__ = "1.0.0"
__all__ = [
    "BLIP2Captioner", 
    "BLIPCaptioner", 
    "VietnameseTranslator",
    "load_image",
    "preprocess_image"
] 