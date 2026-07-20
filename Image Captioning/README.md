# Image Captioning

A beginner-level **applied deep-learning** project that generates natural-language captions for images
using a pretrained Vision-Encoder-Decoder model, transfer learning, no training required.

## Problem Statement
Given an image, produce a one-sentence description of it. We use an off-the-shelf pretrained model
(ViT image encoder → GPT-2 text decoder) and run inference on sample images.

## Dataset / Model
- **Model**: [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) (Hugging Face), a ViT encoder + GPT-2 decoder trained on COCO captions.
- **Sample images** (`data/`): 6 everyday photos (dog, cat, pizza, bicycle) + two scikit-learn sample
  images (a cityscape and a flower).

> Note: the checklist points at the full COCO dataset (large, for *training* a captioner). This project
> instead demonstrates **transfer learning**, using a heavyweight pretrained vision-language model
> off-the-shelf, which is how captioning is done in practice without a captioned training set + GPUs.

## Project Structure
```
Image Captioning/
├── 01_captioning.ipynb   # Load pretrained model, caption sample images, show results
├── utils.py · requirements.txt · README.md
└── data/  (6 sample images + captions.json)
```

*(First run downloads ~1 GB of model weights; loads in ~1 to 2 minutes on CPU.)*

## Key Findings
All captions produced by executing the notebook, not assumed.
- The pretrained ViT-GPT2 model produces **fluent, mostly-accurate captions with zero training**, e.g.:
  - dog → *"a white dog standing in a grassy field"*
  - pizza → *"a pizza in a box on a table"*
  - cat → *"a cat that is looking at something"*
  - flower → *"a flower in a vase on a sunny day"*
- **The encoder, decoder pattern**: the ViT turns the image into patch embeddings; the GPT-2 decoder attends
  to those and generates the caption token-by-token (beam search picks the most likely sentence).
- **Honest failure modes**: it describes what it recognises from its COCO-style training distribution, so it
  can miss context, e.g. a dense cityscape captioned as *"a tall building with a clock on top"*.
- **The takeaway**: this is transfer learning at its most useful, a large vision-language model applied
  off-the-shelf in a few lines, for a task that would otherwise need a big captioned dataset to train.

## Tech Stack
- transformers, torch, pillow, matplotlib

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_captioning.ipynb
```
