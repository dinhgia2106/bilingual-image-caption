"""
Bilingual Image Captioning App

A Gradio-based web application for generating bilingual (English & Vietnamese) 
image captions using state-of-the-art vision-language models.
"""

import gradio as gr
import os
import time
import logging
from PIL import Image
from typing import Optional, Tuple, Dict

# Import our custom modules
from model import BLIP2Captioner, BLIPCaptioner, VietnameseTranslator
from model.utils import load_image, get_image_info, setup_logging

# Setup logging
setup_logging(log_file="app.log", level="INFO")
logger = logging.getLogger(__name__)

# Global variables for models (lazy loading)
current_model = None
current_model_name = None
translator = None

# Available models configuration
AVAILABLE_MODELS = {
    "BLIP-2 OPT-2.7B (Recommended)": {
        "model_name": "Salesforce/blip2-opt-2.7b",
        "class": BLIP2Captioner,
        "description": "Best overall performance, state-of-the-art results"
    },
    "BLIP-2 Flan-T5-XL": {
        "model_name": "Salesforce/blip2-flan-t5-xl", 
        "class": BLIP2Captioner,
        "description": "Good for instruction following and detailed captions"
    },
    "BLIP Large": {
        "model_name": "Salesforce/blip-image-captioning-large",
        "class": BLIPCaptioner,
        "description": "Stable baseline, good performance"
    },
    "BLIP Base (Fast)": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "class": BLIPCaptioner,
        "description": "Lightweight and fast, good for quick testing"
    }
}

# Sample images
SAMPLE_IMAGES = [
    "data/000000000139.jpg",
    "data/000000000285.jpg", 
    "data/000000000632.jpg",
    "data/000000000724.jpg",
    "data/000000000776.jpg"
]


def initialize_translator():
    """Initialize the Vietnamese translator"""
    global translator
    if translator is None:
        try:
            translator = VietnameseTranslator(primary_service="google")
            logger.info("Vietnamese translator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize translator: {e}")
            translator = None
    return translator


def load_model(model_choice: str, use_8bit: bool = False) -> bool:
    """
    Load the selected model
    
    Args:
        model_choice: Selected model name
        use_8bit: Whether to use 8-bit quantization
        
    Returns:
        True if successful, False otherwise
    """
    global current_model, current_model_name
    
    try:
        if model_choice not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_choice}")
        
        # Check if model is already loaded
        if current_model_name == model_choice and current_model is not None:
            logger.info(f"Model {model_choice} already loaded")
            return True
        
        # Load new model
        model_config = AVAILABLE_MODELS[model_choice]
        model_class = model_config["class"]
        model_name = model_config["model_name"]
        
        logger.info(f"Loading model: {model_choice}")
        
        # Initialize model with 8-bit option for BLIP-2
        if model_class == BLIP2Captioner and use_8bit:
            current_model = model_class(model_name, load_in_8bit=True)
        else:
            current_model = model_class(model_name)
        
        current_model_name = model_choice
        logger.info(f"Successfully loaded {model_choice}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model {model_choice}: {e}")
        current_model = None
        current_model_name = None
        return False


def generate_captions(image: Image.Image, 
                     model_choice: str,
                     custom_prompt: str,
                     use_8bit: bool = False,
                     max_length: int = 50,
                     num_beams: int = 3) -> Tuple[str, str, str, Dict]:
    """
    Generate bilingual captions for the given image
    
    Args:
        image: PIL Image
        model_choice: Selected model name
        custom_prompt: Optional custom prompt
        use_8bit: Whether to use 8-bit quantization
        max_length: Maximum caption length
        num_beams: Number of beams for generation
        
    Returns:
        Tuple of (english_caption, vietnamese_caption, status_message, image_info)
    """
    try:
        # Validate inputs
        if image is None:
            return "", "", "Please upload an image first", {}
        
        # Load model if needed
        if not load_model(model_choice, use_8bit):
            return "", "", f"Failed to load model: {model_choice}", {}
        
        # Initialize translator if needed
        trans = initialize_translator()
        if trans is None:
            return "", "", "Failed to initialize translator", {}
        
        # Get image info
        try:
            image_info = {
                "format": image.format or "Unknown",
                "mode": image.mode,
                "size": f"{image.width} x {image.height}",
                "width": image.width,
                "height": image.height
            }
        except:
            image_info = {}
        
        # Generate English caption
        start_time = time.time()
        
        prompt = custom_prompt.strip() if custom_prompt.strip() else None
        english_caption = current_model.generate_caption(
            image=image,
            prompt=prompt,
            max_length=max_length,
            num_beams=num_beams
        )
        
        generation_time = time.time() - start_time
        
        if not english_caption or english_caption == "Unable to generate caption":
            return "", "", "Failed to generate caption", image_info
        
        # Translate to Vietnamese
        start_time = time.time()
        vietnamese_caption = trans.translate(english_caption)
        translation_time = time.time() - start_time
        
        # Create status message
        status_msg = f"Success! Generated in {generation_time:.2f}s, translated in {translation_time:.2f}s"
        
        logger.info(f"Generated caption: {english_caption}")
        logger.info(f"Vietnamese translation: {vietnamese_caption}")
        
        return english_caption, vietnamese_caption, status_msg, image_info
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return "", "", error_msg, {}


def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .model-info {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Bilingual Image Captioning") as app:
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🖼️ Bilingual Image Captioning</h1>
                <p>Generate image captions in English and Vietnamese using state-of-the-art AI models</p>
            </div>
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                # Image input
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                # Sample images
                with gr.Row():
                    gr.Markdown("**Or choose a sample:**")
                
                with gr.Row():
                    sample_buttons = []
                    for i, sample_path in enumerate(SAMPLE_IMAGES[:3]):
                        if os.path.exists(sample_path):
                            btn = gr.Button(f"Sample {i+1}", size="sm")
                            sample_buttons.append((btn, sample_path))
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="BLIP-2 OPT-2.7B (Recommended)",
                    label="Select Model",
                    info="Choose the captioning model"
                )
                
                # Model info display
                model_info = gr.HTML(
                    value=f'<div class="model-info"><b>Model:</b> {AVAILABLE_MODELS["BLIP-2 OPT-2.7B (Recommended)"]["description"]}</div>'
                )
                
                # Advanced options
                with gr.Accordion("⚙️ Advanced Options", open=False):
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt (Optional)",
                        placeholder="e.g., 'a photography of', 'describe this image in detail'",
                        info="Add a custom prompt to guide caption generation"
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Max Length"
                        )
                        
                        num_beams = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Num Beams"
                        )
                    
                    use_8bit = gr.Checkbox(
                        label="Use 8-bit quantization (BLIP-2 only)",
                        value=False,
                        info="Reduces memory usage but may affect quality"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🚀 Generate Captions",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Results")
                
                # Status message
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to generate captions"
                )
                
                # Captions output
                english_output = gr.Textbox(
                    label="🇺🇸 English Caption",
                    interactive=False,
                    lines=3
                )
                
                vietnamese_output = gr.Textbox(
                    label="🇻🇳 Vietnamese Caption", 
                    interactive=False,
                    lines=3
                )
                
                # Image info
                with gr.Accordion("📊 Image Information", open=False):
                    image_info_output = gr.JSON(
                        label="Image Details"
                    )
        
        # Event handlers
        def update_model_info(model_choice):
            if model_choice in AVAILABLE_MODELS:
                desc = AVAILABLE_MODELS[model_choice]["description"]
                return f'<div class="model-info"><b>Model:</b> {desc}</div>'
            return ""
        
        def load_sample_image(sample_path):
            try:
                return Image.open(sample_path)
            except Exception as e:
                logger.error(f"Failed to load sample image {sample_path}: {e}")
                return None
        
        # Connect events
        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        # Sample image buttons
        for btn, sample_path in sample_buttons:
            btn.click(
                fn=lambda path=sample_path: load_sample_image(path),
                outputs=[image_input]
            )
        
        # Generate button
        generate_btn.click(
            fn=generate_captions,
            inputs=[
                image_input,
                model_dropdown, 
                custom_prompt,
                use_8bit,
                max_length,
                num_beams
            ],
            outputs=[
                english_output,
                vietnamese_output,
                status_output,
                image_info_output
            ]
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
                <p><b>About:</b> This application uses state-of-the-art vision-language models (BLIP, BLIP-2) 
                combined with translation services to generate bilingual image captions.</p>
                <p><b>Models:</b> Salesforce BLIP/BLIP-2 | <b>Translation:</b> Deep Translator | <b>Framework:</b> Gradio</p>
            </div>
        """)
    
    return app


def main():
    """Main function to run the application"""
    logger.info("Starting Bilingual Image Captioning App")
    
    # Create the Gradio interface
    app = create_gradio_interface()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main() 