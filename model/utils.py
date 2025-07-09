"""
Utility Functions

This module contains utility functions for image processing, evaluation metrics,
and other helper functions.
"""

import os
import requests
import numpy as np
from PIL import Image, ImageOps
import logging
from typing import Union, List, Tuple, Optional
import json
import csv
from urllib.parse import urlparse
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_source: Union[str, Image.Image]) -> Image.Image:
    """
    Load image from various sources (file path, URL, or PIL Image)
    
    Args:
        image_source: File path, URL, or PIL Image
        
    Returns:
        PIL Image in RGB format
    """
    if isinstance(image_source, Image.Image):
        return image_source.convert('RGB')
    
    elif isinstance(image_source, str):
        # Check if it's a URL
        if image_source.startswith(('http://', 'https://')):
            return load_image_from_url(image_source)
        # Otherwise treat as file path
        elif os.path.exists(image_source):
            try:
                image = Image.open(image_source)
                return image.convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load image from {image_source}: {e}")
                raise
        else:
            raise FileNotFoundError(f"Image file not found: {image_source}")
    
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """
    Load image from URL
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image in RGB format
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        image = Image.open(response.raw)
        return image.convert('RGB')
        
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        raise


def preprocess_image(image: Image.Image, 
                    max_size: Optional[Tuple[int, int]] = None,
                    maintain_aspect_ratio: bool = True) -> Image.Image:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        max_size: Maximum size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if max_size specified
    if max_size:
        if maintain_aspect_ratio:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(max_size, Image.Resampling.LANCZOS)
    
    # Auto-orient based on EXIF data
    image = ImageOps.exif_transpose(image)
    
    return image


def save_results_to_csv(results: List[dict], filename: str):
    """
    Save captioning results to CSV file
    
    Args:
        results: List of dictionaries containing results
        filename: Output CSV filename
    """
    if not results:
        logger.warning("No results to save")
        return
    
    fieldnames = results[0].keys()
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")
        raise


def load_results_from_csv(filename: str) -> List[dict]:
    """
    Load captioning results from CSV file
    
    Args:
        filename: CSV filename
        
    Returns:
        List of dictionaries containing results
    """
    try:
        results = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)
        
        logger.info(f"Loaded {len(results)} results from {filename}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results from CSV: {e}")
        raise


def create_sample_dataset(images_dir: str, output_file: str):
    """
    Create sample dataset from images directory
    
    Args:
        images_dir: Directory containing images
        output_file: Output CSV file path
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = []
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    # Create dataset entries
    dataset = []
    for i, image_file in enumerate(sorted(image_files)):
        dataset.append({
            'image_id': i + 1,
            'image_path': os.path.join(images_dir, image_file),
            'image_name': image_file,
            'english_caption': '',
            'vietnamese_caption': ''
        })
    
    # Save to CSV
    save_results_to_csv(dataset, output_file)
    logger.info(f"Created sample dataset with {len(dataset)} images")


def validate_image_path(image_path: str) -> bool:
    """
    Validate if image path exists and is a valid image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get basic information about an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    if not validate_image_path(image_path):
        raise ValueError(f"Invalid image path: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            return {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': os.path.getsize(image_path)
            }
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        raise


def batch_process_images(image_paths: List[str], 
                        output_dir: str,
                        max_size: Optional[Tuple[int, int]] = None) -> List[str]:
    """
    Batch process multiple images
    
    Args:
        image_paths: List of image file paths
        output_dir: Output directory for processed images
        max_size: Maximum size for resizing
        
    Returns:
        List of processed image paths
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_paths = []
    
    for image_path in image_paths:
        try:
            # Load and preprocess image
            image = load_image(image_path)
            processed_image = preprocess_image(image, max_size)
            
            # Save processed image
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            processed_image.save(output_path)
            
            processed_paths.append(output_path)
            logger.info(f"Processed {filename}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
    
    return processed_paths


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate text
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        BLEU score (0-100)
    """
    try:
        from sacrebleu import sentence_bleu
        
        # Convert to lowercase and tokenize
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # Calculate BLEU score
        score = sentence_bleu(cand_tokens, [ref_tokens])
        return score.score
        
    except ImportError:
        logger.warning("sacrebleu not available, using simple word overlap")
        return calculate_word_overlap(reference, candidate)
    except Exception as e:
        logger.error(f"Failed to calculate BLEU score: {e}")
        return 0.0


def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Calculate word overlap percentage between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap percentage (0-100)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return (len(intersection) / len(union)) * 100 if union else 0.0


def setup_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    ) 