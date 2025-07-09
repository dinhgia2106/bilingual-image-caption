"""
Image Captioning Models

This module contains wrapper classes for BLIP-2 and BLIP models
to generate image captions with consistent interfaces.
"""

import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
import logging
from typing import Union, List, Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCaptioner(ABC):
    """Abstract base class for image captioning models"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def generate_caption(self, image: Union[Image.Image, str], 
                        prompt: Optional[str] = None) -> str:
        """Generate caption for the given image"""
        pass


class BLIP2Captioner(BaseCaptioner):
    """BLIP-2 model wrapper for image captioning"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", 
                 device: str = "auto", load_in_8bit: bool = False):
        super().__init__(model_name, device)
        self.load_in_8bit = load_in_8bit
        self.load_model()
    
    def load_model(self):
        """Load BLIP-2 model and processor"""
        try:
            logger.info(f"Loading BLIP-2 model: {self.model_name}")
            
            # Load processor
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            
            # Load model with optional 8-bit quantization
            if self.load_in_8bit and self.device == "cuda":
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
                logger.info("Model loaded with 8-bit quantization")
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model.to(self.device)
                logger.info(f"Model loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise
    
    def generate_caption(self, image: Union[Image.Image, str], 
                        prompt: Optional[str] = None,
                        max_length: int = 50,
                        min_length: int = 10,
                        num_beams: int = 3,
                        repetition_penalty: float = 1.2) -> str:
        """
        Generate caption for the given image using BLIP-2
        
        Args:
            image: PIL Image or path to image file
            prompt: Optional text prompt for conditional captioning
            max_length: Maximum length of generated caption
            min_length: Minimum length of generated caption  
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated caption as string
        """
        try:
            # Prepare inputs
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            if prompt:
                inputs = self.processor(image, text=prompt, return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to device if needed
            if not self.load_in_8bit:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.device == "cuda":
                    inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v 
                             for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    do_sample=False,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up caption if it includes the prompt
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return "Unable to generate caption"


class BLIPCaptioner(BaseCaptioner):
    """BLIP model wrapper for image captioning"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", 
                 device: str = "auto"):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        """Load BLIP model and processor"""
        try:
            logger.info(f"Loading BLIP model: {self.model_name}")
            
            # Load processor and model
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            
            logger.info(f"BLIP model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise
    
    def generate_caption(self, image: Union[Image.Image, str], 
                        prompt: Optional[str] = None,
                        max_length: int = 50,
                        min_length: int = 10,
                        num_beams: int = 3,
                        repetition_penalty: float = 1.2) -> str:
        """
        Generate caption for the given image using BLIP
        
        Args:
            image: PIL Image or path to image file
            prompt: Optional text prompt for conditional captioning
            max_length: Maximum length of generated caption
            min_length: Minimum length of generated caption
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated caption as string
        """
        try:
            # Prepare inputs
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            if prompt:
                inputs = self.processor(image, text=prompt, return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device == "cuda":
                inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    do_sample=False,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up caption if it includes the prompt
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return "Unable to generate caption"


def get_available_models() -> dict:
    """Return dictionary of available models with their descriptions"""
    return {
        "blip2-opt-2.7b": {
            "model_name": "Salesforce/blip2-opt-2.7b",
            "class": BLIP2Captioner,
            "description": "BLIP-2 with OPT-2.7B language model - Best overall performance",
            "size": "2.7B parameters"
        },
        "blip2-flan-t5-xl": {
            "model_name": "Salesforce/blip2-flan-t5-xl", 
            "class": BLIP2Captioner,
            "description": "BLIP-2 with Flan-T5-XL language model - Good for instruction following",
            "size": "3.9B parameters"
        },
        "blip-large": {
            "model_name": "Salesforce/blip-image-captioning-large",
            "class": BLIPCaptioner,
            "description": "BLIP Large - Stable and reliable baseline",
            "size": "470M parameters"
        },
        "blip-base": {
            "model_name": "Salesforce/blip-image-captioning-base",
            "class": BLIPCaptioner,
            "description": "BLIP Base - Lightweight and fast",
            "size": "247M parameters"
        }
    } 