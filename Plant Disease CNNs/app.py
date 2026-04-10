import torch
torch.set_num_threads(2)
import gradio as gr
import matplotlib.pyplot as plt
import io
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import timm
import json
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolve asset paths relative to this file
_BASE_DIR = Path(__file__).resolve().parent

# Load class names
with open(_BASE_DIR / "class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Create model
# Create model EXACTLY like training
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_names)
)

# Load trained weights
model.load_state_dict(
    torch.load(_BASE_DIR / "efficientnet_plant_best.pth", map_location=device),
    strict=True
)

model.to(device)
model.eval()

# IMPORTANT: Match Kaggle validation transform exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image):
    if image is None:
        raise ValueError("No image provided. Please upload an image before predicting.")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probs, 0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    result_text = f"### {predicted_class} ({confidence:.2f}%)"

    # Create figure with a small top area for text and larger area for image
    fig = plt.figure(figsize=(4, 5))
    gs = fig.add_gridspec(2, height_ratios=[0.15, 0.85])

    ax_text = fig.add_subplot(gs[0])
    ax_img = fig.add_subplot(gs[1])

    ax_text.text(0.5, 0.5, f"{predicted_class} ({confidence:.2f}%)",
                 ha="center", va="center", fontsize=14, weight="bold")
    ax_text.axis("off")

    ax_img.imshow(image)
    ax_img.axis("off")

    fig.tight_layout(pad=0)

    # Convert Matplotlib figure to PIL Image for Gradio
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    annotated_img = Image.open(buf).convert("RGB")

    return annotated_img, result_text

with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Plant Disease Classification")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", height=300, width=300)

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil")
            output_label = gr.Markdown()

    btn = gr.Button("Predict")

    btn.click(predict, inputs=input_image, outputs=[output_image, output_label])

demo.launch()