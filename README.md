# Bilingual Image Captioning

This project delivers a high-quality, bilingual (English & Vietnamese) image captioning application. The core model was selected through a data-driven process, involving systematic testing and BLEU score evaluation of multiple state-of-the-art vision-language models.

---

## Key Features

- **Data-Driven Model Selection**
  The primary model, **GIT Large COCO**, was chosen based on superior performance in comparative tests.

- **High-Quality Captions**
  Generates detailed, context-aware descriptions of images.

- **Bilingual Support**
  Provides captions in both English and Vietnamese using a robust translation layer.

- **Flexible & Modular**
  Easily switch between different models (e.g., quality-focused vs. speed-focused) via the interface.

- **Interactive Web UI**
  A simple and intuitive Gradio interface for easy testing and demonstration.

---

## Model Evaluation and Selection

To ensure the best balance of quality and performance, several popular models were benchmarked against a subset of the COCO dataset. The results guided the final model selection.

### Benchmark Results

| Model                 | Avg. Inference Time | BLEU Score | Status  | Key Characteristic                 |
| :-------------------- | :------------------ | :--------- | :------ | :--------------------------------- |
| **GIT Large COCO**    | 5.43s               | 14.14      | Success | Best quality and balance           |
| **GIT Base**          | 1.42s               | 5.82       | Success | Fastest, suitable for real-time    |
| **BLIP-2 Flan-T5-XL** | 6.90s               | 5.79       | Success | Slower, generates generic captions |

### Selected Model

**microsoft/git-large-coco**

**Reasoning:**

- **Superior Quality**
  Achieved a BLEU score of **14.14**, more than double its competitors, indicating significantly more accurate and human-like captions.

- **Rich, Contextual Descriptions**
  Unlike other models producing generic phrases, **GIT-Large** excels at identifying objects and their relationships.
  Example:

  - _GIT-Large:_ "a large brown bear laying on top of a green field."
  - _Other models:_ "the bear is brown"

- **Balanced Performance**
  Delivers top-tier quality with a reasonable inference time (**5.43s**), making it the most balanced and reliable option.

---

## Getting Started

### Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/dinhgia2106/bilingual-image-caption.git
cd bilingual-image-caption
pip install -r requirements.txt
```

### Running the Application

Launch the Gradio web application.

```bash
python app.py
```

Then open a web browser and navigate to:

```
http://127.0.0.1:7860
```

You can upload your own image or use the provided examples.

---

## Production Deployment Recommendations

### Performance and Speed

- **GPU Acceleration**
  Deploy on a server with a CUDA-enabled GPU.

- **Model Quantization**
  Convert the model to 8-bit or 16-bit precision (FP16/INT8) to improve speed and reduce VRAM usage with minimal quality loss.

- **Batch Processing**
  Modify the application to process images in batches for high throughput environments.

- **Model Caching**
  Keep the model loaded in memory to avoid reload delays between requests.

### Cost Optimization

- **Translation Service**
  Replace `deep-translator` with a dedicated or self-hosted translation API for production.

- **Right-Sized Models**
  For speed-critical use cases, offer the `GIT Base` model, which is approximately four times faster.

- **Batch Translation**
  When handling multiple captions, send them in a single API call to reduce overhead and cost.

### Scalability and Infrastructure

- **Containerization**
  Package the application with Docker for consistent deployments.

- **Load Balancing**
  Use a load balancer (e.g., Nginx) to distribute traffic across multiple application instances.

- **API Gateway**
  Place the service behind an API Gateway to handle rate limiting, authentication, and caching.

- **Monitoring**
  Integrate tools such as Prometheus and Grafana to monitor inference time, error rates, and resource utilization.

---

## Technology Stack

- **Core Models:**
  `microsoft/git-large-coco`, `microsoft/git-base`, `Salesforce/blip2-flan-t5-xl`

- **ML/DL Framework:**
  `PyTorch`

- **Hugging Face Suite:**
  `transformers`, `accelerate`

- **Web Interface:**
  `Gradio`

- **Translation:**
  `deep-translator` (Google Translate backend)

- **Evaluation:**
  `NLTK` (for BLEU score calculation)
