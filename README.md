# Neural Style Transfer with VGG19 - PyTorch + Streamlit

> Turn any photo into a painting using Gatys et al. Neural Style Transfer. Built with VGG19, PyTorch and an interactive Streamlit UI.

![Demo](data/output/demo.png)

### What this project does
Upload a **Content Image** (your photo) and a **Style Image** (Starry Night, fluid art, etc.) and generate a new image that keeps the structure of content but painted with the texture, colors and brush strokes of style.

This implementation goes beyond a basic tutorial. It includes manual LBFGS optimization, proper Gram matrix normalization, and a full debug of loss explosion, NaN handling and color shifts.

### Features
- **VGG19 Feature Extractor** - Uses conv4_2 for content, conv1_1 to conv5_1 for style
- **Dual Optimizers** - Adam for quick preview, LBFGS for sharp final results
- **Interactive UI** - Built with Streamlit, live loss chart, image resizing, save to disk
- **Stable Training** - Gradient clipping, history_size tuning, correct deprocessing to avoid pink noise
- **Loss Tracking** - Total, content and style loss visualized in real time

### How it Works
**Content Loss:**
```
L_content = MSE( F_l(content), F_l(generated) ) at conv4_2
```

**Style Loss via Gram Matrix:**
```
Gram(F) = (F * F^T) / (H * W)
L_style = sum over style layers MSE( Gram(F_style), Gram(F_generated) )
Total = alpha * L_content + beta * L_style
```
Gram captures texture by correlating feature maps. Dividing by H*W not C*H*W gives better scale for artistic transfer.

### Project Structure
```
.
├── main.py                 # Streamlit app
├── src/
│   ├── model/
│   │   ├── vgg.py          # VGG19FeatureExtractor
│   │   ├── losses.py       # content_loss, style_loss, gram_matrix
│   │   └── transfer_engine.py  # StyleTransferEngine with LBFGS
│   └── utils/
│       └── image_processing.py # preprocess_vgg, deprocess_vgg
├── data/
│   ├── input/
│   └── output/
└── requirements.txt
```

### Installation
```bash
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Requirements: `torch, torchvision, streamlit, pillow`

### Usage
```bash
streamlit run main.py
```
1. Upload Content Image
2. Upload Style Image
3. Tune parameters in sidebar
4. Click Stylize

Outputs saved to `data/output/stylized_lbfgs_TIMESTAMP.jpg`

### Recommended Hyperparameters

For **Artistic Transfer** (Starry Night, Van Gogh):

| Parameter | Value | Why |
| --- | --- | --- |
| Content Weight | 1.0 to 100 | Keeps structure |
| Style Weight | 3000 to 10000 | Controls painting strength |
| LBFGS lr | 0.3 to 0.8 | LBFGS needs higher lr than Adam |
| Steps | 200 to 300 | LBFGS does 3-5 inner iters per step |
| Max Size | 512 | Balance quality vs VRAM |

For **Simple Content** (single tree on hill):
Use Content 1000, Style 100 to avoid content loss.

If total loss at Step 10 is > 5000, lower style weight. Healthy range is 500 to 3000. If loss spikes to 45M, you are exploding.

### Challenges Solved
This project looked simple but had real research bugs:

**1. Loss Explosion to NaN:** With Gram / (C*H*W) and beta 1e12, loss went 77k -> 32k -> NaN. Reason: Gram gradient scales cubically with features. Fix: normalize Gram by H*W, use beta 1e3 to 1e4, clip grad to [-1,1].

**2. LBFGS Hessian Corruption:** Using `generated.clamp_()` inside LBFGS broke its history of s and y vectors. Fix: use `generated.data.clamp_()` and remove strong_wolfe line search, set max_iter=3, history_size=5.

**3. Pink Noise Output:** Blue style gave pink output. Reason: pixels went to -3.0, deprocess without final clamp to [0,1]. Fix: deprocess as `t * std + mean` then `clamp(0,1)` before PIL conversion.

**4. Spiky Loss Chart:** LBFGS jumps, not walks. Loss 606 -> 186 -> 1.6M -> 566 is expected when old Hessian is discarded. Reduced by lowering lr to 0.3.

### Results
| Content | Style | Output |
| --- | --- | --- |
| Tree field | Blue fluid | Tree shape preserved with fluid swirls |
| Portrait | Starry Night | Face with Van Gogh strokes |

Add your own results in `data/output/`

### Future Improvements
- [ ] Preserve color option (only transfer texture)
- [ ] Fast feed-forward style transfer with AdaIN
- [ ] Video style transfer
- [ ] Masked style transfer for selective regions

### Author
Built by Saransh Basu. Learning PyTorch optimization deep dive.

### Reference
Gatys, L. A., et al. "A Neural Algorithm of Artistic Style." CVPR 2016.
