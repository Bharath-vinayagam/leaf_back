# 🌿 Leaf Disease Detector

> A deep learning-powered tool to detect plant leaf diseases using images — built with TensorFlow and Streamlit.

![License](https://img.shields.io/github/license/cavxn/Leaf-Disease-Detector)
![Stars](https://img.shields.io/github/stars/cavxn/Leaf-Disease-Detector?style=social)

---

## 🧠 Project Overview

This project uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify plant leaf diseases across **38 categories**, including:

- Apple Scab, Black Rot, Cedar Rust
- Grape Black Rot, Esca, Leaf Blight
- Tomato Leaf Mold, Early Blight, Septoria
- Potato Early/Late Blight
- Healthy class for each plant

---

## 🚀 Features

- 🔍 Upload an image of a leaf
- 🧠 Deep learning model predicts the disease
- ✅ Displays top prediction with confidence
- 🌐 Clean, interactive UI with Streamlit

---

## 📦 Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras**
- **NumPy, PIL**
- **Streamlit** for UI

---

## 📷 Sample Output

![Sample](samples/sample_output.jpg)  
*(Image showing prediction result with confidence)*

---

## 📁 Project Structure
leaf-disease-detector/
├── app.py / streamlit_app.py # Main Streamlit app
├── final_leaf_disease_model.keras
├── samples/ # 200 sample images for test
├── requirements.txt
├── .gitignore
└── README.md

yaml

## How to Run Locally

### 1. Clone the repo
git clone https://github.com/cavxn/Leaf-Disease-Detector.git
cd Leaf-Disease-Detector

2. Create and activate environment (optional)
python3 -m venv plantenv
source plantenv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py
The app will open in your browser at http://localhost:8501

Try it Online
Coming soon: Streamlit Share Link Here
(Optional: Add your live hosted link here)

Model Info
Trained on: PlantVillage Dataset
Input size: 224x224 RGB
Accuracy: ~98% on test set

 Acknowledgements
Dataset: PlantVillage, via Kaggle
UI: Streamlit
Model: TensorFlow/keras

License
This project is licensed under the MIT License — see the LICENSE file for details.

yaml

