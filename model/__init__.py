"""
Bilingual Image Captioning Model Package

This package contains models and utilities for generating bilingual 
(English & Vietnamese) image captions using state-of-the-art vision-language models.
"""

from .captioning_models import BLIP2Captioner, BLIPCaptioner, GITCaptioner, get_available_models, create_captioner
from .translator import VietnameseTranslator
from .utils import (
    load_image, preprocess_image, save_results_to_csv, load_results_from_csv,
    calculate_bleu_score, calculate_cider_score, calculate_rouge_l, evaluate_captions
)

__version__ = "1.0.0"
__all__ = [
    "BLIP2Captioner", 
    "BLIPCaptioner", 
    "GITCaptioner",
    "VietnameseTranslator",
    "load_image",
    "preprocess_image",
    "get_available_models",
    "create_captioner",
    "save_results_to_csv",
    "load_results_from_csv",
    "calculate_bleu_score",
    "calculate_cider_score", 
    "calculate_rouge_l",
    "evaluate_captions"
] 