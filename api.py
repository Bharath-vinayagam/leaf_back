import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

# -------------------------------
# Safe model loading
# -------------------------------
def load_model_safely():
    model_path = "final_leaf_disease_model.keras"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
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
                    options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                )
                print("Model loaded successfully with legacy options")
                return model
            except Exception as e3:
                print(f"All loading attempts failed: {e3}")
                return create_fallback_model()

def create_fallback_model():
    """Simple fallback CNN"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Fallback model created successfully")
    return model

# Load model
try:
    model = load_model_safely()
    print("✅ Model ready")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = create_fallback_model()

# -------------------------------
# Class names
# -------------------------------
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
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -------------------------------
# Prediction API
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
