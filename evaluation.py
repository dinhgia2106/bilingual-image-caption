"""
Bilingual Image Captioning Evaluation Script

This script evaluates the performance of different captioning models
and provides quality assessment for both English and Vietnamese captions.
"""

import os
import json
import time
import logging
from PIL import Image
from typing import List, Dict, Tuple

from model import BLIP2Captioner, BLIPCaptioner, VietnameseTranslator
from model.utils import setup_logging, calculate_bleu_score

# Setup logging
setup_logging(log_file="evaluation.log", level="INFO")
logger = logging.getLogger(__name__)

# Evaluation configuration
EVALUATION_CONFIG = {
    "models_to_test": [
        {"name": "BLIP-2 OPT-2.7B", "model_name": "Salesforce/blip2-opt-2.7b", "class": BLIP2Captioner},
        {"name": "BLIP Large", "model_name": "Salesforce/blip-image-captioning-large", "class": BLIPCaptioner},
    ],
    "test_images": [
        "data/000000000139.jpg",
        "data/000000000285.jpg", 
        "data/000000000632.jpg",
        "data/000000000724.jpg",
        "data/000000000776.jpg"
    ],
    "generation_params": {
        "max_length": 50,
        "num_beams": 3
    }
}


def load_ground_truth_captions(json_path: str) -> Dict[str, List[str]]:
    """Load ground truth captions from COCO dataset file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create mapping from image_id to captions
        captions_map = {}
        for annotation in data.get('annotations', []):
            image_id = annotation['image_id']
            caption = annotation['caption'].strip()
            
            if image_id not in captions_map:
                captions_map[image_id] = []
            captions_map[image_id].append(caption)
        
        # Map filenames to captions
        filename_to_captions = {}
        for image_info in data.get('images', []):
            filename = image_info['file_name']
            image_id = image_info['id']
            
            if image_id in captions_map:
                filename_to_captions[filename] = captions_map[image_id]
        
        return filename_to_captions
        
    except Exception as e:
        logger.error(f"Error loading ground truth captions: {e}")
        return {}


def evaluate_single_image(image_path: str, 
                         model: object,
                         translator: VietnameseTranslator,
                         ground_truth_captions: List[str] = None) -> Dict:
    """Evaluate captioning performance on a single image"""
    
    try:
        # Load image
        image = Image.open(image_path)
        image_name = os.path.basename(image_path)
        
        # Generate caption
        start_time = time.time()
        english_caption = model.generate_caption(
            image=image,
            max_length=EVALUATION_CONFIG["generation_params"]["max_length"],
            num_beams=EVALUATION_CONFIG["generation_params"]["num_beams"]
        )
        generation_time = time.time() - start_time
        
        # Translate to Vietnamese
        start_time = time.time()
        vietnamese_caption = translator.translate(english_caption)
        translation_time = time.time() - start_time
        
        # Calculate BLEU score if ground truth available
        bleu_score = None
        if ground_truth_captions and english_caption:
            bleu_score = calculate_bleu_score(english_caption, ground_truth_captions)
        
        result = {
            "image_name": image_name,
            "image_path": image_path,
            "english_caption": english_caption,
            "vietnamese_caption": vietnamese_caption,
            "generation_time": round(generation_time, 2),
            "translation_time": round(translation_time, 2),
            "total_time": round(generation_time + translation_time, 2),
            "bleu_score": round(bleu_score, 4) if bleu_score else None,
            "ground_truth_captions": ground_truth_captions,
            "caption_length": len(english_caption.split()) if english_caption else 0
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating image {image_path}: {e}")
        return {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "error": str(e),
            "english_caption": "",
            "vietnamese_caption": "",
            "generation_time": 0,
            "translation_time": 0,
            "total_time": 0,
            "bleu_score": None
        }


def evaluate_model_performance(model_config: Dict, 
                             test_images: List[str],
                             ground_truth: Dict[str, List[str]]) -> Dict:
    """Evaluate performance of a specific model"""
    
    logger.info(f"Starting evaluation for model: {model_config['name']}")
    
    try:
        # Initialize model
        model_class = model_config["class"]
        model_name = model_config["model_name"]
        model = model_class(model_name)
        
        # Initialize translator
        translator = VietnameseTranslator(primary_service="google")
        
        results = []
        total_generation_time = 0
        total_translation_time = 0
        bleu_scores = []
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Get ground truth for this image
            image_name = os.path.basename(image_path)
            gt_captions = ground_truth.get(image_name, [])
            
            # Evaluate image
            result = evaluate_single_image(image_path, model, translator, gt_captions)
            results.append(result)
            
            # Accumulate metrics
            if "error" not in result:
                total_generation_time += result["generation_time"]
                total_translation_time += result["translation_time"]
                
                if result["bleu_score"] is not None:
                    bleu_scores.append(result["bleu_score"])
            
            # Log progress
            logger.info(f"Processed {image_name}: {result.get('english_caption', 'ERROR')[:50]}...")
        
        # Calculate summary statistics
        num_successful = len([r for r in results if "error" not in r])
        avg_generation_time = total_generation_time / num_successful if num_successful > 0 else 0
        avg_translation_time = total_translation_time / num_successful if num_successful > 0 else 0
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
        
        model_summary = {
            "model_name": model_config["name"],
            "model_path": model_config["model_name"],
            "total_images": len(test_images),
            "successful_evaluations": num_successful,
            "failed_evaluations": len(test_images) - num_successful,
            "avg_generation_time": round(avg_generation_time, 2),
            "avg_translation_time": round(avg_translation_time, 2),
            "avg_total_time": round(avg_generation_time + avg_translation_time, 2),
            "avg_bleu_score": round(avg_bleu_score, 4) if avg_bleu_score else None,
            "bleu_scores_count": len(bleu_scores),
            "detailed_results": results
        }
        
        logger.info(f"Completed evaluation for {model_config['name']}")
        logger.info(f"Success rate: {num_successful}/{len(test_images)}")
        logger.info(f"Average generation time: {avg_generation_time:.2f}s")
        logger.info(f"Average BLEU score: {avg_bleu_score:.4f}" if avg_bleu_score else "No BLEU scores available")
        
        return model_summary
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_config['name']}: {e}")
        return {
            "model_name": model_config["name"],
            "error": str(e),
            "total_images": len(test_images),
            "successful_evaluations": 0,
            "failed_evaluations": len(test_images)
        }


def generate_evaluation_report(results: List[Dict], output_file: str = "evaluation_report.json"):
    """Generate comprehensive evaluation report"""
    
    # Create summary
    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models_tested": len(results),
        "test_configuration": EVALUATION_CONFIG,
        "model_results": results
    }
    
    # Add comparative analysis
    successful_models = [r for r in results if "error" not in r and r["successful_evaluations"] > 0]
    
    if successful_models:
        # Find best performing model by BLEU score
        models_with_bleu = [r for r in successful_models if r.get("avg_bleu_score") is not None]
        if models_with_bleu:
            best_bleu_model = max(models_with_bleu, key=lambda x: x["avg_bleu_score"])
            summary["best_model_by_bleu"] = {
                "name": best_bleu_model["model_name"],
                "bleu_score": best_bleu_model["avg_bleu_score"]
            }
        
        # Find fastest model
        fastest_model = min(successful_models, key=lambda x: x["avg_total_time"])
        summary["fastest_model"] = {
            "name": fastest_model["model_name"],
            "avg_time": fastest_model["avg_total_time"]
        }
    
    # Save report
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation report: {e}")
    
    return summary


def print_summary_results(results: List[Dict]):
    """Print summary of evaluation results to console"""
    
    print("\n" + "="*80)
    print("BILINGUAL IMAGE CAPTIONING - EVALUATION RESULTS")
    print("="*80)
    
    for result in results:
        if "error" in result:
            print(f"\nModel: {result['model_name']}")
            print(f"Status: FAILED - {result['error']}")
            continue
        
        print(f"\nModel: {result['model_name']}")
        print(f"Success Rate: {result['successful_evaluations']}/{result['total_images']}")
        print(f"Average Generation Time: {result['avg_generation_time']}s")
        print(f"Average Translation Time: {result['avg_translation_time']}s")
        print(f"Average Total Time: {result['avg_total_time']}s")
        
        if result.get('avg_bleu_score'):
            print(f"Average BLEU Score: {result['avg_bleu_score']:.4f}")
        else:
            print("BLEU Score: Not available")
        
        # Show sample captions
        if result.get("detailed_results"):
            print("\nSample Captions:")
            for i, detail in enumerate(result["detailed_results"][:2], 1):
                if "error" not in detail:
                    print(f"  {i}. {detail['image_name']}")
                    print(f"     EN: {detail['english_caption']}")
                    print(f"     VI: {detail['vietnamese_caption']}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function"""
    logger.info("Starting bilingual image captioning evaluation")
    
    # Load ground truth captions
    ground_truth_file = "data/captions_filtered.json"
    ground_truth = {}
    
    if os.path.exists(ground_truth_file):
        ground_truth = load_ground_truth_captions(ground_truth_file)
        logger.info(f"Loaded ground truth for {len(ground_truth)} images")
    else:
        logger.warning(f"Ground truth file not found: {ground_truth_file}")
    
    # Check test images exist
    available_images = [img for img in EVALUATION_CONFIG["test_images"] if os.path.exists(img)]
    logger.info(f"Found {len(available_images)} test images")
    
    if not available_images:
        logger.error("No test images found. Please check image paths.")
        return
    
    # Evaluate each model
    all_results = []
    for model_config in EVALUATION_CONFIG["models_to_test"]:
        try:
            result = evaluate_model_performance(model_config, available_images, ground_truth)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_config['name']}: {e}")
            all_results.append({
                "model_name": model_config["name"],
                "error": str(e),
                "total_images": len(available_images),
                "successful_evaluations": 0
            })
    
    # Generate and save report
    report = generate_evaluation_report(all_results)
    
    # Print summary to console
    print_summary_results(all_results)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main() 