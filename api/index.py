from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# Create FastAPI app instance that Vercel will import
app = FastAPI()


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Leaf backend is running",
        "endpoints": ["/predict", "/detect-leaf", "/docs"],
    }


# -------------------------------
# Safe model loading (copied from root api.py with minor path tweaks)
# -------------------------------
def _model_path() -> str:
    # Resolve relative to this file so it works on Vercel
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base_dir, "final_leaf_disease_model.keras")
    return candidate


def load_model_safely():
    model_path = _model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully with compile=False")
        return model
    except Exception as e1:
        print(f"First loading attempt failed: {e1}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
            print("Model loaded successfully with custom_objects")
            return model
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'),
                )
                print("Model loaded successfully with legacy options")
                return model
            except Exception as e3:
                print(f"Third loading attempt failed: {e3}")
                try:
                    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
                    print("Model loaded successfully with safe_mode=True")
                    return model
                except Exception as e4:
                    print(f"Fourth loading attempt failed: {e4}")
                    print("Using fallback model - your trained model may have compatibility issues")
                    return create_fallback_model()


def create_fallback_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Fallback model created successfully")
    return model


def is_leaf_image(image_array):
    try:
        hsv = tf.image.rgb_to_hsv(image_array)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mean_saturation = tf.reduce_mean(s)
        mean_value = tf.reduce_mean(v)

        green_mask = tf.logical_and(tf.greater(h, 0.2), tf.less(h, 0.4))
        green_ratio = tf.reduce_mean(tf.cast(green_mask, tf.float32))

        gray = tf.image.rgb_to_grayscale(image_array)
        texture_variance = tf.math.reduce_variance(gray)

        print(
            f"Debug - Green ratio: {green_ratio:.3f}, Saturation: {mean_saturation:.3f}, Value: {mean_value:.3f}, Texture: {texture_variance:.5f}"
        )

        leaf_score = 0
        if green_ratio > 0.25:
            leaf_score += 3
        elif green_ratio > 0.15:
            leaf_score += 2
        elif green_ratio > 0.05:
            leaf_score += 1

        if 0.08 < mean_saturation < 0.95:
            leaf_score += 2
        if 0.05 < mean_value < 0.95:
            leaf_score += 2
        if texture_variance > 0.005:
            leaf_score += 2
        if texture_variance > 0.02:
            leaf_score += 1
        if texture_variance > 0.04:
            leaf_score += 1
        if green_ratio < 0.03:
            leaf_score -= 1

        return leaf_score >= 4, leaf_score / 11.0
    except Exception as e:
        print(f"Error in leaf detection: {e}")
        return True, 0.5


def advanced_leaf_detection(image_array):
    try:
        gray = tf.image.rgb_to_grayscale(image_array)
        edges_x = tf.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        edges_y = tf.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
        edge_density = tf.reduce_mean(edges_x) + tf.reduce_mean(edges_y)

        height, width = 224, 224
        aspect_ratio = height / width

        non_zero_pixels = tf.reduce_sum(tf.cast(tf.greater(gray, 0.1), tf.float32))
        coverage_ratio = non_zero_pixels / (224 * 224)

        texture_variance = tf.math.reduce_variance(gray)

        score = 0
        if edge_density > 0.015:
            score += 2
        if 0.25 < aspect_ratio < 4.0:
            score += 1
        if coverage_ratio > 0.15:
            score += 2
        if texture_variance > 0.02:
            score += 1
        if texture_variance > 0.04:
            score += 1

        return score >= 2, score / 7.0
    except Exception as e:
        print(f"Error in advanced leaf detection: {e}")
        return True, 0.5


# Load model once at import time (cold start)
try:
    disease_model = load_model_safely()
    print("✅ Disease classification model ready")
except Exception as e:
    print(f"❌ Error loading disease model: {e}")
    disease_model = create_fallback_model()


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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
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

        preds = disease_model.predict(img_array)
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

        img_array = np.array(img) / 255.0
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


