
import streamlit as st
import torch
from PIL import Image
import os
import datetime
# Make src importable
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.model.vgg import VGG19FeatureExtractor
from src.model.transfer_engine import StyleTransferEngine
from src.utils.image_processing import preprocess_vgg, deprocess_vgg, load_image
import torch.nn.functional as F
from src.model.losses import gram_matrix

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

@st.cache_resource
def load_extractor(content_layer, style_layers):
    return VGG19FeatureExtractor(content_layer=content_layer, style_layers=style_layers)

st.title("Neural Style Transfer - VGG19")

with st.sidebar:
    st.header("Parameters")
    max_size = st.slider("Max Image Size", 128, 768, 512, step=64)
    content_weight = st.number_input("Content Weight (alpha)", value=1.0)
    style_weight = st.number_input("Style Weight (beta)", value=1e6, format="%.0e")
    num_steps = st.slider("Steps", 50, 1000, 300, step=50)
    lr = st.slider("Learning Rate", 0.001, 0.1, 0.02, format="%.3f")
    device_str = st.selectbox("Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
    st.info(f"Torch device: {device_str}")

col1, col2, col3 = st.columns(3)

with col1:
    content_file = st.file_uploader("Content Image", type=["jpg", "png", "jpeg"])
    if content_file:
        content_pil = Image.open(content_file).convert("RGB")
        st.image(content_pil, caption="Content", use_column_width=True)

with col2:
    style_file = st.file_uploader("Style Image", type=["jpg", "png", "jpeg"])
    if style_file:
        style_pil = Image.open(style_file).convert("RGB")
        st.image(style_pil, caption="Style", use_column_width=True)

with col3:
    st.write("Output will appear here")

if content_file and style_file:
    if st.button("Stylize", type="primary"):
        device = torch.device(device_str)

        # Resize keeping aspect ratio
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

        with st.spinner("Loading VGG19..."):
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

        st.write("Optimizing pixels...")
        progress = st.progress(0)
        loss_chart = st.empty()

        # Manual loop to show progress because engine.run is closed loop
        # For now call engine.run and update at end
        generated = content_tensor.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([generated], lr=lr)
        history = []
        history_total = []
        history_content = []
        history_style = []

        progress = st.progress(0)
        chart_total = st.empty()
        chart_split = st.empty()
        for step in range(num_steps):
            optimizer.zero_grad()
            feats = engine.extractor.extract_features(generated)

            c_loss = F.mse_loss(feats[content_layer], engine.content_target)

            s_loss = 0
            for layer in style_layers:
                gen_gram = gram_matrix(feats[layer])
                s_loss += F.mse_loss(gen_gram, engine.style_targets[layer])
            s_loss = s_loss / len(style_layers)

            total = content_weight * c_loss + style_weight * s_loss
            total.backward()
            optimizer.step()

            with torch.no_grad():
                generated.clamp_(-2.5, 2.8)

            history_total.append(total.item())
            history_content.append(c_loss.item())
            history_style.append(s_loss.item())

            if step % 10 == 0:
                progress.progress((step + 1) / num_steps)
                chart_total.line_chart({"total_loss": history_total})
                chart_split.line_chart({
                    "content_loss": history_content,
                    "style_loss": history_style
                })

        final_tensor = generated.detach()
        output_pil = deprocess_vgg(final_tensor)

        with col3:
            st.image(output_pil, caption=f"Stylized {num_steps} steps", use_column_width=True)
            st.line_chart(history)

        # Save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"stylized_{timestamp}.jpg"
        out_path = os.path.join("data/output", out_name)
        os.makedirs("data/output", exist_ok=True)
        output_pil.save(out_path)
        st.success(f"Saved to {out_path}")
else:
    st.warning("Upload both content and style images")