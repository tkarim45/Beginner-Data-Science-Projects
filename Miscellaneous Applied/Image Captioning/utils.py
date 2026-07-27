"""
Image Captioning — Utility Functions
Generate natural-language captions for images with a pretrained
Vision-Encoder-Decoder model (ViT encoder + GPT-2 decoder), nlpconnect/vit-gpt2.
"""
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

def load_model():
    """Load the pretrained captioning model + processor + tokenizer."""
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, processor, tokenizer

def caption_image(path, model, processor, tokenizer, max_length=16, num_beams=4):
    """Return a generated caption string for the image at `path`."""
    img = Image.open(path).convert("RGB")
    pixels = processor(images=[img], return_tensors="pt").pixel_values
    with torch.no_grad():
        out = model.generate(pixels, max_length=max_length, num_beams=num_beams)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()
