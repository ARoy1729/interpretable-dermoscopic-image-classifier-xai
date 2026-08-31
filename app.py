from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import APP_TITLE, CHECKPOINT_PATH, CLASS_DESCRIPTIONS, CLASS_NAMES, IMAGE_SIZE
from src.explainability import GradCAM, overlay_heatmap, preprocess_pil, predict, shap_attribution
from src.model import build_model_from_checkpoint

st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading DenseNet121 checkpoint…")
def load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train the model first with scripts/train.py."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = build_model_from_checkpoint(checkpoint, device)

    # A neutral background in ImageNet-normalized space is deliberately small: SHAP's
    # expected-gradient computation is the slower part of this demo, so a handful of
    # deterministic baselines keeps the app responsive while preserving the comparison.
    background = torch.zeros((6, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    background += torch.randn_like(background) * 0.03
    background = background.clamp(-0.2, 0.2)
    background[0].zero_()
    background = background.to(device)
    return model, device, class_names, background


def explanation_text(prediction, gradcam, shap_map):
    grad_focus = float((gradcam > 0.55).mean())
    shap_focus = float((shap_map > 0.55).mean())
    focus_phrase = "the central lesion and its border" if grad_focus < 0.35 else "a relatively broad region around the visible lesion"
    agreement = "Both methods emphasize similar areas." if abs(grad_focus - shap_focus) < 0.12 else "The two methods emphasize somewhat different pixels, which is useful when auditing model behavior."
    return (
        f"The model predicts **{prediction.label}** ({CLASS_DESCRIPTIONS.get(prediction.label, prediction.label)}) "
        f"with **{prediction.confidence:.1%} confidence**. Grad-CAM mainly highlights {focus_phrase}. "
        f"SHAP highlights pixels that most increase the predicted class score. {agreement} "
        "These explanations describe what this trained model used; they are not a medical diagnosis."
    )


st.title("DermExplain")
st.caption("Explainable HAM10000 skin-lesion classification with DenseNet121 + Grad-CAM + SHAP")
st.write(
    "Upload a dermoscopic image to see the model's seven-class prediction and two complementary views of where the network found useful evidence."
)

with st.sidebar:
    st.subheader("Model")
    st.write("DenseNet121 • ImageNet transfer learning")
    st.write("Frozen backbone + trained 7-class head")
    st.write("7 HAM10000 classes")
    st.write("Weighted cross-entropy during training")
    st.caption("For research/portfolio demonstration only. Not for clinical decision-making.")

uploaded = st.file_uploader("Choose a skin-lesion image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image above to start the explanation.")
    st.stop()

try:
    model, device, class_names, background = load_runtime()
    image = Image.open(uploaded).convert("RGB")
    input_tensor = preprocess_pil(image).to(device)

    with st.spinner("Generating prediction and explanations…"):
        prediction = predict(model, input_tensor, class_names, device)
        gradcam = GradCAM(model)
        try:
            cam = gradcam(input_tensor, prediction.index)
        finally:
            gradcam.close()
        shap_map = shap_attribution(model, input_tensor, prediction.index, background, device)
        grad_overlay = overlay_heatmap(image, cam)
        shap_overlay = overlay_heatmap(image, shap_map)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted class", prediction.label)
    with col2:
        st.metric("Confidence", f"{prediction.confidence:.1%}")

    st.markdown(explanation_text(prediction, cam, shap_map))

    st.subheader("Model attention")
    image_col, grad_col, shap_col = st.columns(3)
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with grad_col:
        st.image(grad_overlay, caption="Grad-CAM overlay", use_container_width=True)
    with shap_col:
        st.image(shap_overlay, caption="SHAP overlay", use_container_width=True)

    st.subheader("Class probabilities")
    probability_rows = [
        {"class": name, "probability": float(probability)}
        for name, probability in zip(class_names, prediction.probabilities)
    ]
    probability_rows = sorted(probability_rows, key=lambda row: row["probability"], reverse=True)
    st.bar_chart({row["class"]: row["probability"] for row in probability_rows})

except Exception as exc:
    st.error(f"The image could not be processed: {exc}")
    st.exception(exc)
