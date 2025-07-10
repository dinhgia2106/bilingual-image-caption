"""
Bilingual Image Captioning Application

A Gradio-based web application for generating bilingual (English & Vietnamese) 
image captions using state-of-the-art vision-language models.
"""

import gradio as gr
import os
import time
import logging
from PIL import Image
from typing import Tuple, Optional

from model import GITCaptioner, VietnameseTranslator, get_available_models, create_captioner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BilingualCaptioningApp:
    def __init__(self):
        self.captioner = None
        self.translator = None
        self.current_model = None
        self.load_default_model()
        
    def load_default_model(self):
        """Load the default best performing model"""
        try:
            logger.info("Loading default model: GIT Large COCO")
            self.captioner = GITCaptioner("microsoft/git-large-coco")
            self.translator = VietnameseTranslator()
            self.current_model = "GIT Large COCO"
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            self.captioner = None
            self.translator = VietnameseTranslator()
            
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        try:
            if model_name == self.current_model:
                return f"Model {model_name} is already loaded"
                
            logger.info(f"Switching to model: {model_name}")
            
            model_mapping = {
                "GIT Large COCO": ("git_large", "microsoft/git-large-coco"),
                "GIT Base": ("git_base", "microsoft/git-base"), 
                "BLIP-2 Flan-T5-XL": ("blip2_flan_t5", "Salesforce/blip2-flan-t5-xl")
            }
            
            if model_name in model_mapping:
                model_key, model_id = model_mapping[model_name]
                self.captioner = create_captioner(model_key)
                self.current_model = model_name
                return f"Successfully switched to {model_name}"
            else:
                return f"Unknown model: {model_name}"
                
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return f"Error switching to {model_name}: {str(e)}"
    
    def generate_captions(self, image: Image.Image, model_name: str = "GIT Large COCO", 
                         custom_prompt: str = "") -> Tuple[str, str, str]:
        """
        Generate bilingual captions for an image
        
        Args:
            image: PIL Image
            model_name: Model to use for captioning
            custom_prompt: Optional custom prompt
            
        Returns:
            Tuple of (english_caption, vietnamese_caption, status_message)
        """
        if image is None:
            return "", "", "Please upload an image first"
        
        try:
            # Switch model if needed
            if model_name != self.current_model:
                switch_msg = self.switch_model(model_name)
                if "Error" in switch_msg:
                    return "", "", switch_msg
            
            if self.captioner is None:
                return "", "", "Model not loaded. Please try restarting the application."
            
            start_time = time.time()
            
            # Generate English caption
            english_caption = self.captioner.generate_caption(
                image=image,
                prompt=custom_prompt if custom_prompt.strip() else None,
                max_length=50,
                num_beams=3
            )
            
            # Translate to Vietnamese
            vietnamese_caption = self.translator.translate(english_caption)
            
            generation_time = time.time() - start_time
            
            status_msg = f"Generated in {generation_time:.2f} seconds using {self.current_model}"
            
            return english_caption, vietnamese_caption, status_msg
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "", "", f"Error: {str(e)}"

def create_sample_gallery():
    """Create gallery of sample images"""
    sample_images = []
    data_dir = "data"
    
    if os.path.exists(data_dir):
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in sorted(image_files)[:6]:  # Show first 6 images
            img_path = os.path.join(data_dir, img_file)
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    sample_images.append((img_path, img_file))
            except Exception as e:
                logger.warning(f"Could not load sample image {img_file}: {e}")
    
    return sample_images

def create_interface():
    """Create Gradio interface"""
    app = BilingualCaptioningApp()
    
    # Get available models (working ones only)
    available_models = ["GIT Large COCO", "GIT Base", "BLIP-2 Flan-T5-XL"]
    
    # Create sample gallery
    sample_images = create_sample_gallery()
    
    with gr.Blocks(title="Bilingual Image Captioning", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # Bilingual Image Captioning
        
        Generate high-quality captions in both English and Vietnamese using state-of-the-art AI models.
        Upload your own image or select from samples below.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value="GIT Large COCO",
                    label="Select Model",
                    info="GIT Large COCO is recommended for best quality"
                )
                
                # Custom prompt
                prompt_input = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="e.g., 'A photo of' or 'This image shows'",
                    lines=2
                )
                
                # Generate button
                generate_btn = gr.Button("Generate Captions", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Outputs
                english_output = gr.Textbox(
                    label="English Caption",
                    lines=3,
                    show_copy_button=True
                )
                
                vietnamese_output = gr.Textbox(
                    label="Vietnamese Caption", 
                    lines=3,
                    show_copy_button=True
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=2
                )
        
        # Sample images gallery
        if sample_images:
            gr.Markdown("### Sample Images")
            with gr.Row():
                for img_path, img_name in sample_images:
                    with gr.Column(scale=1):
                        sample_img = gr.Image(
                            value=img_path,
                            label=img_name,
                            height=150,
                            interactive=False
                        )
                        sample_btn = gr.Button(f"Use {img_name}", size="sm")
                        sample_btn.click(
                            fn=lambda path=img_path: Image.open(path),
                            outputs=image_input
                        )
        
        # About section
        with gr.Accordion("About this Application", open=False):
            gr.Markdown("""
            This application uses advanced vision-language models to generate descriptive captions for images.
            
            **Models Available:**
            - **GIT Large COCO**: Best quality, detailed captions (Recommended)
            - **GIT Base**: Faster but simpler captions  
            - **BLIP-2 Flan-T5-XL**: High quality alternative
            
            **Features:**
            - Bilingual output (English + Vietnamese)
            - Multiple AI models to choose from
            - Custom prompts for guided generation
            - Sample images for testing
            
            **Technology:**
            - Models: Microsoft GIT, Salesforce BLIP-2
            - Translation: Google Translate API
            - Interface: Gradio
            """)
        
        # Connect the generate button
        generate_btn.click(
            fn=app.generate_captions,
            inputs=[image_input, model_dropdown, prompt_input],
            outputs=[english_output, vietnamese_output, status_output]
        )
        
        # Example usage
        gr.Examples(
            examples=[
                [sample_images[0][0] if sample_images else None, "GIT Large COCO", ""],
            ] if sample_images else [],
            inputs=[image_input, model_dropdown, prompt_input],
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    # Launch settings
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    ) 