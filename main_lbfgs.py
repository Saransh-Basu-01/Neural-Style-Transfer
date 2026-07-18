import streamlit as st
import torch
from PIL import Image
import os
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.model.vgg import VGG19FeatureExtractor
from src.model.transfer_engine import StyleTransferEngine
from src.utils.image_processing import preprocess_vgg, deprocess_vgg

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

@st.cache_resource
def load_extractor(content_layer, style_layers):
    return VGG19FeatureExtractor(content_layer=content_layer, style_layers=style_layers)

st.title("Neural Style Transfer - VGG19 LBFGS")

with st.sidebar:
    st.header("Parameters")
    max_size = st.slider("Max Image Size", 128, 768, 512, step=64)
    content_weight = st.number_input("Content Weight (alpha)", value=1)
    style_weight = st.number_input("Style Weight (beta)", value=5000.0, format="%.0e")
    num_steps = st.slider("Steps", 20, 500, 200, step=20)
    lr = st.slider("Learning Rate for LBFGS", 0.1, 1.5, 0.8, step=0.1)
    device_str = st.selectbox("Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
    st.info(f"Torch device: {device_str}")

col1, col2, col3 = st.columns(3)

with col1:
    content_file = st.file_uploader("Content Image", type=["jpg", "png", "jpeg"])
    if content_file:
        content_pil = Image.open(content_file).convert("RGB")
        st.image(content_pil, caption="Content", width="stretch")

with col2:
    style_file = st.file_uploader("Style Image", type=["jpg", "png", "jpeg"])
    if style_file:
        style_pil = Image.open(style_file).convert("RGB")
        st.image(style_pil, caption="Style", width="stretch")

with col3:
    st.write("Output will appear here")

if content_file and style_file:
    if st.button("Stylize with LBFGS", type="primary"):
        device = torch.device(device_str)

        def resize_pil(pil_img, max_side):
            w, h = pil_img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                return pil_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            return pil_img

        content_pil_resized = resize_pil(content_pil, max_size)
        style_pil_resized = resize_pil(style_pil, max_size)

        content_tensor = preprocess_vgg(content_pil_resized).to(device)
        style_tensor = preprocess_vgg(style_pil_resized).to(device)

        content_layer = "conv4_2"
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        with st.spinner("Loading VGG19 and preparing targets..."):
            extractor = load_extractor(content_layer, style_layers)
            engine = StyleTransferEngine(
                feature_extractor=extractor,
                device=device,
                content_weight=content_weight,
                style_weight=style_weight,
                content_layer=content_layer,
                style_layers=style_layers
            )
            engine.prepare_targets(content_tensor, style_tensor)

        with st.spinner(f"Optimizing with LBFGS for {num_steps} steps... this is slower per step but sharper"):
            generated_tensor, loss_history = engine.run(
                content_tensor=content_tensor,
                num_steps=num_steps,
                lr=lr
            )

        output_pil = deprocess_vgg(generated_tensor)

        with col3:
            st.image(output_pil, caption=f"Stylized {num_steps} LBFGS steps", width="stretch")
            st.line_chart({"total_loss": loss_history})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"stylized_lbfgs_{timestamp}.jpg"
        out_path = os.path.join("data/output", out_name)
        os.makedirs("data/output", exist_ok=True)
        output_pil.save(out_path)
        st.success(f"Saved to {out_path}")
else:
    st.warning("Upload both content and style images")