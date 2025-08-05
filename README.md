# 🎭 Face Swap + Enhancement (InsightFace + GFPGAN)

This project performs high-quality face swapping using [InsightFace](https://github.com/deepinsight/insightface) and enhances the result using [GFPGAN](https://github.com/TencentARC/GFPGAN). It provides a clean and easy-to-use web interface built with [Gradio](https://www.gradio.app/).

---

## 📦 Features

- 🔁 Swap faces between two images using InsightFace InSwapper (ONNX)
- ✨ Restore and enhance faces using GFPGAN (v1.4)
- 🌐 Simple Gradio-based web interface
- 🧠 Uses ONNXRuntime for face swapping (CPU/GPU compatible)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dhrutitagline/face_swap.git
cd face_swap
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 📥 Download Model Weights

The app automatically downloads the following models on first run:
They are saved inside the models/ folder.

If needed, you can manually run this:

```bash
mkdir -p models
cd models

# Download the models
curl -L -o GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
curl -L -o inswapper_128.onnx https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx
cd ..
```

### 🚀 Run the App
```bash
python app.py
```

### Once the app is running, open your browser and go to:
```bash
http://127.0.0.1:7860
```