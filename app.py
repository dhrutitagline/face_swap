import insightface
import os
import onnxruntime
import cv2
import gfpgan
import tempfile
import time
import gradio as gr
import numpy as np


class Predictor:
    def __init__(self):
        self.setup()

    def setup(self):
        os.makedirs('models', exist_ok=True)
        os.chdir('models')
        if not os.path.exists('GFPGANv1.4.pth'):
            os.system('curl -L -o GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth')

        if not os.path.exists('inswapper_128.onnx'):
           os.system('curl -L -o inswapper_128.onnx https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx')
        os.chdir('..')

        """Load the model into memory to make running multiple predictions efficient"""
        self.face_swapper = insightface.model_zoo.get_model(
            'models/inswapper_128.onnx',
            providers=onnxruntime.get_available_providers()
        )
        self.face_enhancer = gfpgan.GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1)
        self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    def get_face(self, img_data):
        analysed = self.face_analyser.get(img_data)
        try:
            largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest
        except:
            print("No face found")
            return None

    def predict(self, input_image, swap_image):
        try:
            # Convert to OpenCV BGR
            frame = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            swap = cv2.cvtColor(np.array(swap_image), cv2.COLOR_RGB2BGR)

            original_size = frame.shape[:2][::-1]  # (width, height)
            print("Original Image Size:", original_size)

            face = self.get_face(frame)
            source_face = self.get_face(swap)

            swapped = self.face_swapper.get(frame.copy(), face, source_face, paste_back=True)

            _, _, enhanced = self.face_enhancer.enhance(swapped, has_aligned=False, paste_back=True)

            result_size = enhanced.shape[:2][::-1]  # (width, height)
            print("Output Image Size (before resize):", result_size)

            # Resize if needed
            result_resized = cv2.resize(enhanced, original_size, interpolation=cv2.INTER_LANCZOS4)
            print("Output Image Size (after resize):", result_resized.shape[:2][::-1])

            # Save
            out_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(out_path, result_resized)
            return out_path

        except Exception as e:
            print(f"Error: {e}")
            return None



# Instantiate the predictor
predictor = Predictor()

# Modern Gradio Interface
gr.Interface(
    fn=predictor.predict,
    inputs=[
        gr.Image(type="pil", label="Target Image"),
        gr.Image(type="pil", label="Swap Image")
    ],
    outputs=gr.Image(type="filepath", label="Result"),
    title="Swap Faces Using InsightFace + GFPGAN",
    description="Upload a target image and a source image to swap faces. Uses InsightFace + GFPGAN for enhancement.",
    allow_flagging="never",
    examples=None  # You can use actual file paths here if you're hosting
).launch(debug=True)
