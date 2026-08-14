from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Leaf backend is running on Vercel (ONNX engine)",
        "endpoints": ["/predict", "/detect-leaf", "/docs"],
    }

def _model_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base_dir, "final_leaf_disease_model.onnx")
    return candidate

model_path = _model_path()
if os.path.exists(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print("✅ ONNX model loaded successfully")
else:
    raise FileNotFoundError(f"Model file {model_path} not found")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

def is_leaf_image(image_array):
    try:
        img = image_array[0] if len(image_array.shape) == 4 else image_array
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c + 1e-7

        h = np.zeros_like(r)
        mask_g = (max_c == g)
        mask_r = (max_c == r)
        mask_b = (max_c == b)

        h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
        h = h / 6.0

        s = np.where(max_c == 0, 0, delta / (max_c + 1e-7))
        v = max_c

        mean_saturation = np.mean(s)
        mean_value = np.mean(v)

        green_mask = (h > 0.2) & (h < 0.4)
        green_ratio = np.mean(green_mask)

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        texture_variance = np.var(gray)

        leaf_score = 0
        if green_ratio > 0.25: leaf_score += 3
        elif green_ratio > 0.15: leaf_score += 2
        elif green_ratio > 0.05: leaf_score += 1

        if 0.08 < mean_saturation < 0.95: leaf_score += 2
        if 0.05 < mean_value < 0.95: leaf_score += 2
        if texture_variance > 0.005: leaf_score += 2
        if texture_variance > 0.02: leaf_score += 1
        if texture_variance > 0.04: leaf_score += 1
        if green_ratio < 0.03: leaf_score -= 1

        return leaf_score >= 4, leaf_score / 11.0
    except Exception as e:
        print(f"Error in leaf detection: {e}")
        return True, 0.5

def advanced_leaf_detection(image_array):
    try:
        img = image_array[0] if len(image_array.shape) == 4 else image_array
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        edges_x = np.abs(gray[:, 1:] - gray[:, :-1])
        edges_y = np.abs(gray[1:, :] - gray[:-1, :])
        edge_density = np.mean(edges_x) + np.mean(edges_y)

        aspect_ratio = 1.0
        non_zero_pixels = np.sum(gray > 0.1)
        coverage_ratio = non_zero_pixels / (224 * 224)
        texture_variance = np.var(gray)

        score = 0
        if edge_density > 0.015: score += 2
        if 0.25 < aspect_ratio < 4.0: score += 1
        if coverage_ratio > 0.15: score += 2
        if texture_variance > 0.02: score += 1
        if texture_variance > 0.04: score += 1

        return score >= 2, score / 7.0
    except Exception as e:
        print(f"Error in advanced leaf detection: {e}")
        return True, 0.5

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        is_leaf, leaf_confidence = is_leaf_image(img_array)
        if not is_leaf:
            return {
                "is_leaf": False,
                "leaf_confidence": float(leaf_confidence),
                "message": "This image does not appear to be a leaf image. Please upload a clear image of a plant leaf for disease detection.",
                "class": "Not a leaf",
                "confidence": 0.0,
            }

        preds = session.run(None, {input_name: img_array})[0][0]
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        return {
            "is_leaf": True,
            "leaf_confidence": float(leaf_confidence),
            "class": predicted_class,
            "confidence": confidence,
            "message": f"Leaf detected with {leaf_confidence:.2%} confidence. Disease classification: {predicted_class}",
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect-leaf")
async def detect_leaf_only(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        is_leaf_basic, confidence_basic = is_leaf_image(img_array)
        is_leaf_advanced, confidence_advanced = advanced_leaf_detection(img_array)

        final_is_leaf = is_leaf_basic and is_leaf_advanced
        final_confidence = (confidence_basic + confidence_advanced) / 2

        return {
            "is_leaf": bool(final_is_leaf),
            "confidence": float(final_confidence),
            "basic_detection": {"is_leaf": bool(is_leaf_basic), "confidence": float(confidence_basic)},
            "advanced_detection": {"is_leaf": bool(is_leaf_advanced), "confidence": float(confidence_advanced)},
            "message": "Leaf detection completed successfully",
        }
    except Exception as e:
        return {"error": str(e)}
