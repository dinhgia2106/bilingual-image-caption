"""
Bilingual Image Captioning - Model Comparison Script

This script tests and compares the actual performance of:
- BLIP-2 (multiple variants)
- GIT (GenerativeImage2Text) 
- InstructBLIP
- Vintern-1B-v2 (Vietnamese-specific)

Purpose: Find the best model through actual testing, not theoretical comparison.
"""

import os
import json
import time
import logging
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from deep_translator import GoogleTranslator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self):
        self.results = {}
        self.test_images = []
        self.translator = GoogleTranslator(source='en', target='vi')
        self.reference_captions = self.load_reference_captions()
        
        # Model configurations (removed BLIP-2 OPT due to poor performance)
        self.models_to_test = {
            "blip2_flan_t5": {
                "name": "BLIP-2 Flan-T5-XL", 
                "model_id": "Salesforce/blip2-flan-t5-xl",
                "class": "Blip2ForConditionalGeneration",
                "processor": "Blip2Processor"
            },
            "git_large": {
                "name": "GIT Large COCO",
                "model_id": "microsoft/git-large-coco",
                "class": "GitForCausalLM", 
                "processor": "AutoProcessor"
            },
            "git_base": {
                "name": "GIT Base",
                "model_id": "microsoft/git-base",
                "class": "GitForCausalLM",
                "processor": "AutoProcessor"  
            },
            # "instructblip": {
            #     "name": "InstructBLIP Vicuna-7B",
            #     "model_id": "Salesforce/instructblip-vicuna-7b", 
            #     "class": "InstructBlipForConditionalGeneration",
            #     "processor": "InstructBlipProcessor"
            # },
            # "vintern": {
            #     "name": "Vintern-1B-v2 (Vietnamese)",
            #     "model_id": "5CD-AI/Vintern-1B-v2",
            #     "class": "AutoModel", 
            #     "processor": "AutoTokenizer",
            #     "special": "vietnamese_native"
            # }
        }
    
    def load_reference_captions(self):
        """Load reference captions from COCO data"""
        captions_file = "data/captions_filtered.json"
        
        if not os.path.exists(captions_file):
            logger.warning(f"Reference captions not found: {captions_file}")
            return {}
        
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create mapping from image filename to captions
            image_to_captions = {}
            
            # Create filename to ID mapping
            filename_to_id = {}
            for img in data['images']:
                filename_to_id[img['file_name']] = img['id']
            
            # Group captions by image
            for annotation in data['annotations']:
                image_id = annotation['image_id']
                caption = annotation['caption']
                
                # Find corresponding filename
                filename = None
                for fname, img_id in filename_to_id.items():
                    if img_id == image_id:
                        filename = fname
                        break
                
                if filename:
                    if filename not in image_to_captions:
                        image_to_captions[filename] = []
                    image_to_captions[filename].append(caption)
            
            logger.info(f"Loaded reference captions for {len(image_to_captions)} images")
            return image_to_captions
            
        except Exception as e:
            logger.error(f"Failed to load reference captions: {e}")
            return {}
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        try:
            from sacrebleu import sentence_bleu
            score = sentence_bleu(candidate, [reference])
            return score.score
        except:
            # Simple word overlap if sacrebleu not available
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            if not ref_words or not cand_words:
                return 0.0
            intersection = len(ref_words.intersection(cand_words))
            return (intersection / len(ref_words.union(cand_words))) * 100
    
    def calculate_metrics(self, model_results: Dict) -> Dict:
        """Calculate BLEU and other metrics for model results"""
        if model_results.get("status") != "success":
            return {}
        
        metrics = {
            'bleu_scores': [],
            'reference_found': 0,
            'total_images': len(model_results['results'])
        }
        
        for result in model_results['results']:
            image_name = result['image']
            candidate = result['english_caption'].strip()
            
            # Get reference caption
            if self.reference_captions and image_name in self.reference_captions:
                # Use first reference caption
                reference = self.reference_captions[image_name][0].strip()
                metrics['reference_found'] += 1
            else:
                # No reference available, skip BLEU calculation
                continue
            
            # Calculate BLEU
            bleu_score = self.calculate_bleu_score(reference, candidate)
            metrics['bleu_scores'].append(bleu_score)
        
        # Calculate averages
        if metrics['bleu_scores']:
            metrics['bleu_avg'] = sum(metrics['bleu_scores']) / len(metrics['bleu_scores'])
        else:
            metrics['bleu_avg'] = 0.0
        
        return metrics
        
    def load_test_images(self):
        """Load sample images for testing"""
        # Use existing data images
        data_dir = "data"
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:10]:  # Test with 10 images
            img_path = os.path.join(data_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                self.test_images.append({
                    "path": img_path,
                    "name": img_file,
                    "image": image
                })
                logger.info(f"Loaded test image: {img_file}")
            except Exception as e:
                logger.error(f"Failed to load {img_file}: {e}")
    
    def test_blip2_model(self, model_config: Dict, test_images: List) -> Dict:
        """Test BLIP-2 variants"""
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            
            logger.info(f"Loading {model_config['name']}...")
            processor = Blip2Processor.from_pretrained(model_config['model_id'])
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_config['model_id'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            results = []
            total_time = 0
            
            for img_data in test_images:
                start_time = time.time()
                
                # Generate caption
                inputs = processor(images=img_data["image"], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_length=50,
                        min_length=5,
                        num_beams=5,
                        do_sample=False,
                        early_stopping=True,
                        repetition_penalty=1.2
                    )
                    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Translate to Vietnamese
                try:
                    vietnamese_caption = self.translator.translate(caption)
                except:
                    vietnamese_caption = "Translation failed"
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                results.append({
                    "image": img_data["name"],
                    "english_caption": caption,
                    "vietnamese_caption": vietnamese_caption,
                    "generation_time": generation_time
                })
                
                logger.info(f"  {img_data['name']}: {caption}")
            
            return {
                "model_name": model_config['name'],
                "total_time": total_time,
                "avg_time": total_time / len(test_images),
                "results": results,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to test {model_config['name']}: {e}")
            return {
                "model_name": model_config['name'], 
                "status": "failed",
                "error": str(e)
            }
    
    def test_git_model(self, model_config: Dict, test_images: List) -> Dict:
        """Test GIT variants"""
        try:
            from transformers import GitForCausalLM, AutoProcessor
            
            logger.info(f"Loading {model_config['name']}...")
            processor = AutoProcessor.from_pretrained(model_config['model_id'])
            model = GitForCausalLM.from_pretrained(
                model_config['model_id'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            results = []
            total_time = 0
            
            for img_data in test_images:
                start_time = time.time()
                
                # Generate caption  
                inputs = processor(images=img_data["image"], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_length=50)
                    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Translate to Vietnamese
                try:
                    vietnamese_caption = self.translator.translate(caption)
                except:
                    vietnamese_caption = "Translation failed"
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                results.append({
                    "image": img_data["name"],
                    "english_caption": caption,
                    "vietnamese_caption": vietnamese_caption,
                    "generation_time": generation_time
                })
                
                logger.info(f"  {img_data['name']}: {caption}")
            
            return {
                "model_name": model_config['name'],
                "total_time": total_time,
                "avg_time": total_time / len(test_images),
                "results": results,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to test {model_config['name']}: {e}")
            return {
                "model_name": model_config['name'],
                "status": "failed", 
                "error": str(e)
            }
    
    def test_instructblip_model(self, model_config: Dict, test_images: List) -> Dict:
        """Test InstructBLIP model"""
        try:
            from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
            
            logger.info(f"Loading {model_config['name']}...")
            processor = InstructBlipProcessor.from_pretrained(model_config['model_id'])
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_config['model_id'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            results = []
            total_time = 0
            
            for img_data in test_images:
                start_time = time.time()
                
                # Generate caption with instruction
                prompt = "Describe this image in detail."
                inputs = processor(images=img_data["image"], text=prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_length=50,
                        min_length=5,
                        num_beams=3,
                        do_sample=False
                    )
                    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Clean up instruction from output
                if caption and prompt in caption:
                    caption = caption.replace(prompt, "").strip()
                
                # Ensure caption is not empty
                if not caption or caption.strip() == "":
                    caption = "Unable to generate caption"
                
                # Translate to Vietnamese
                try:
                    vietnamese_caption = self.translator.translate(caption)
                except:
                    vietnamese_caption = "Translation failed"
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                results.append({
                    "image": img_data["name"],
                    "english_caption": caption,
                    "vietnamese_caption": vietnamese_caption,  
                    "generation_time": generation_time
                })
                
                logger.info(f"  {img_data['name']}: {caption}")
            
            return {
                "model_name": model_config['name'],
                "total_time": total_time,
                "avg_time": total_time / len(test_images),
                "results": results,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to test {model_config['name']}: {e}")
            return {
                "model_name": model_config['name'],
                "status": "failed",
                "error": str(e)
            }
    
    def test_vintern_model(self, model_config: Dict, test_images: List) -> Dict:
        """Test Vintern Vietnamese model"""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            
            logger.info(f"Loading {model_config['name']}...")
            
            # Try to load config first to check if it exists
            try:
                config = AutoConfig.from_pretrained(model_config['model_id'], trust_remote_code=True)
                logger.info(f"Config loaded successfully: {config.__class__.__name__}")
            except Exception as config_error:
                logger.error(f"Failed to load config: {config_error}")
                raise Exception(f"Config loading failed: {config_error}")
            
            # Load tokenizer and model with error handling
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['model_id'], 
                trust_remote_code=True,
                padding_side="left"
            )
            
            model = AutoModel.from_pretrained(
                model_config['model_id'],
                config=config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            results = []
            total_time = 0
            
            for img_data in test_images:
                start_time = time.time()
                
                # Vietnamese prompt
                question = "Mô tả hình ảnh một cách chi tiết."
                
                # Process image (simplified - may need adjustment)
                try:
                    response, _ = model.chat(tokenizer, img_data["image"], question, 
                                           generation_config=dict(max_new_tokens=100, do_sample=False))
                    
                    generation_time = time.time() - start_time
                    total_time += generation_time
                    
                    results.append({
                        "image": img_data["name"],
                        "english_caption": "N/A (Vietnamese native)",
                        "vietnamese_caption": response,
                        "generation_time": generation_time
                    })
                    
                    logger.info(f"  {img_data['name']}: {response}")
                    
                except Exception as e:
                    logger.error(f"Error processing {img_data['name']}: {e}")
                    results.append({
                        "image": img_data["name"],
                        "english_caption": "Error",
                        "vietnamese_caption": f"Error: {str(e)}",
                        "generation_time": time.time() - start_time
                    })
            
            return {
                "model_name": model_config['name'],
                "total_time": total_time,
                "avg_time": total_time / len(test_images),
                "results": results,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to test {model_config['name']}: {e}")
            return {
                "model_name": model_config['name'],
                "status": "failed",
                "error": str(e)
            }
    
    def run_comparison(self):
        """Run the complete model comparison"""
        logger.info("Starting Model Comparison...")
        
        # Load test images
        self.load_test_images()
        if not self.test_images:
            logger.error("No test images found!")
            return
        
        logger.info(f"Testing with {len(self.test_images)} images")
        
        # Test each model
        for model_key, model_config in self.models_to_test.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing: {model_config['name']}")
            logger.info(f"{'='*50}")
            
            try:
                # Clear GPU memory before each model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Route to appropriate test function
                if "blip2" in model_key:
                    result = self.test_blip2_model(model_config, self.test_images)
                elif "git" in model_key:
                    result = self.test_git_model(model_config, self.test_images)
                elif "instructblip" in model_key:
                    result = self.test_instructblip_model(model_config, self.test_images)
                elif "vintern" in model_key:
                    result = self.test_vintern_model(model_config, self.test_images)
                else:
                    result = {"model_name": model_config['name'], "status": "not_implemented"}
                
                # Calculate metrics for successful results
                if result.get("status") == "success":
                    metrics = self.calculate_metrics(result)
                    result.update(metrics)  # Add metrics to result
                
                self.results[model_key] = result
                
                # Log summary
                if result.get("status") == "success":
                    logger.info(f" {model_config['name']} - Avg time: {result['avg_time']:.2f}s")
                    if 'bleu_avg' in result:
                        logger.info(f"   BLEU Score: {result['bleu_avg']:.2f} (refs: {result.get('reference_found', 0)}/{result.get('total_images', 0)})")
                else:
                    logger.info(f" {model_config['name']} - Failed: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Critical error testing {model_config['name']}: {e}")
                self.results[model_key] = {
                    "model_name": model_config['name'],
                    "status": "critical_error", 
                    "error": str(e)
                }
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save comparison results to JSON"""
        output_file = "model_comparison_results.json"
        
        # Make results serializable
        serializable_results = {}
        for key, result in self.results.items():
            serializable_result = result.copy()
            # Remove non-serializable items if any
            serializable_results[key] = serializable_result
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print comparison summary"""
        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        
        successful_models = []
        failed_models = []
        
        for key, result in self.results.items():
            if result.get("status") == "success":
                successful_models.append(result)
            else:
                failed_models.append(result)
        
        # Sort successful models by average time
        successful_models.sort(key=lambda x: x.get("avg_time", float('inf')))
        
        logger.info("\n SUCCESSFUL MODELS (sorted by speed):")
        for i, model in enumerate(successful_models, 1):
            logger.info(f"{i}. {model['model_name']}")
            logger.info(f"   Average time: {model['avg_time']:.2f}s")
            if 'bleu_avg' in model:
                logger.info(f"   BLEU Score: {model['bleu_avg']:.2f}")
            logger.info(f"   Memory usage: {model.get('memory_usage', 0) / 1024**2:.1f}MB")
        
        if failed_models:
            logger.info("\n FAILED MODELS:")
            for model in failed_models:
                logger.info(f"   {model['model_name']}: {model.get('error', 'Unknown error')}")
        
        # Recommendations
        logger.info(f"\n RECOMMENDATIONS:")
        if successful_models:
            fastest = successful_models[0]
            logger.info(f"   Fastest: {fastest['model_name']} ({fastest['avg_time']:.2f}s)")
            
            if len(successful_models) > 1:
                logger.info(f"   Alternative: {successful_models[1]['model_name']} ({successful_models[1]['avg_time']:.2f}s)")
        
        logger.info(f"\n Total models tested: {len(self.results)}")
        logger.info(f" Successful: {len(successful_models)}")
        logger.info(f" Failed: {len(failed_models)}")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import torch
        from transformers import AutoModel
        from deep_translator import GoogleTranslator
    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.info("Please install: pip install torch transformers deep-translator")
        exit(1)
    
    # Run comparison
    comparison = ModelComparison()
    comparison.run_comparison() 