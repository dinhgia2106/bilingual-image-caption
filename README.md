# Bilingual Image Captioning

A state-of-the-art web application for generating bilingual (English & Vietnamese) image captions using advanced vision-language models.

## Features

- **Multiple AI Models**: Support for BLIP-2 and BLIP models with different sizes
- **Bilingual Output**: Automatic translation to Vietnamese using multiple translation services
- **Professional Web Interface**: Clean Gradio-based UI with advanced options
- **Memory Optimization**: 8-bit quantization support for resource-constrained environments
- **Customizable Generation**: Custom prompts, beam search, and length control
- **Sample Images**: Pre-loaded COCO dataset samples for quick testing

## Supported Models

- **BLIP-2 OPT-2.7B** (Recommended): Best overall performance
- **BLIP-2 Flan-T5-XL**: Excellent for instruction following
- **BLIP Large**: Stable baseline with good performance
- **BLIP Base**: Lightweight and fast for quick testing

## Installation

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
python app.py
```

The application will launch at `http://localhost:7860`

### Web Interface Features

1. **Image Upload**: Upload any image for captioning
2. **Model Selection**: Choose from 4 different pre-trained models
3. **Advanced Options**:
   - Custom prompts for guided generation
   - Adjustable max length (20-100 tokens)
   - Beam search configuration (1-5 beams)
   - 8-bit quantization for memory efficiency

### Programmatic Usage

```python
from model import BLIP2Captioner, VietnameseTranslator
from PIL import Image

# Initialize models
captioner = BLIP2Captioner("Salesforce/blip2-opt-2.7b")
translator = VietnameseTranslator()

# Load image
image = Image.open("path/to/image.jpg")

# Generate caption
english_caption = captioner.generate_caption(image)
vietnamese_caption = translator.translate(english_caption)

print(f"English: {english_caption}")
print(f"Vietnamese: {vietnamese_caption}")
```

## Project Structure

```
bilingual-image-caption/
├── app.py                    # Main Gradio application
├── model/                    # Core model package
│   ├── __init__.py
│   ├── captioning_models.py  # BLIP/BLIP-2 model wrappers
│   ├── translator.py         # Vietnamese translation service
│   └── utils.py             # Utility functions
├── data/                     # Sample images and test data
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## Model Architecture

### Captioning Models

- **BLIP2Captioner**: Wrapper for BLIP-2 models with OPT/Flan-T5 backends
- **BLIPCaptioner**: Wrapper for original BLIP models
- Support for custom prompts and generation parameters
- Automatic device detection and memory optimization

### Translation Service

- **Primary Service**: Google Translate via deep-translator
- **Fallback Services**: MyMemory, Linguee for reliability
- Vietnamese-specific text processing and error handling
- Retry logic for robust translation

## Development

### Adding New Models

1. Create model class inheriting from `BaseCaptioner`
2. Implement `generate_caption()` method
3. Add to `AVAILABLE_MODELS` in `app.py`

### Adding Translation Languages

1. Extend `VietnameseTranslator` class
2. Add language-specific post-processing
3. Update UI language options
