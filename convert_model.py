import tensorflow as tf
import tf2onnx
import onnx
import os

keras_path = "final_leaf_disease_model.keras"
onnx_path = "final_leaf_disease_model.onnx"

print("Loading Keras model...")
model = tf.keras.models.load_model(keras_path, compile=False)
print("Model loaded successfully.")

@tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input_1")])
def model_func(x):
    return model(x)

print("Converting model to ONNX format...")
onnx_model, _ = tf2onnx.convert.from_function(
    model_func,
    input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input_1")],
    opset=13
)

onnx.save(onnx_model, onnx_path)
print(f"✅ Successfully converted and saved ONNX model to {onnx_path}!")
print(f"ONNX Model File Size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
