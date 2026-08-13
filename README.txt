================================================================================
                           LEAF DISEASE DETECTOR API
================================================================================

--------------------------------------------------------------------------------
SHORT DESCRIPTION
--------------------------------------------------------------------------------
Leaf Disease Detector API is an end-to-end deep learning computer vision pipeline 
and REST backend designed to classify plant leaf diseases across 38 distinct 
categories. The system incorporates a pre-screening heuristic validation pipeline 
(color space, texture variance, and edge analysis) to verify that an uploaded image 
is an actual plant leaf before running classification inference. The core deep 
learning model is optimized using ONNX Runtime for serverless deployment on Vercel.

--------------------------------------------------------------------------------
DATASET SOURCE AND LICENSING
--------------------------------------------------------------------------------
- Dataset Name: PlantVillage Dataset
- Primary Source: Kaggle / GitHub (<DATASET_URL>, e.g., https://www.kaggle.com/datasets/emware/plantvillage-dataset)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Citation Note: When using this dataset or model in academic or commercial work, 
  please cite the original authors of the PlantVillage dataset (Hughes & Salathé, 2015).
- Placeholders: Replace <DATASET_URL> with the exact repository link used for training.

--------------------------------------------------------------------------------
DATASET DETAILS
--------------------------------------------------------------------------------
- Total Samples: Approximately 54,300 RGB images across 38 classes.
- Target Variable: Categorical plant-disease label (e.g., 'Tomato___healthy', 
  'Potato___Early_blight', 'Apple___Apple_scab', etc.).
- Image Resolution: Original resolutions vary; normalized to 224x224 RGB.
- Preprocessing Pipeline:
  * Resizing: Standardized image dimensions to 224x224 pixels.
  * Normalization: Pixel intensity scaling to the [0.0, 1.0] range (`img / 255.0`).
  * Data Augmentation (during training): Random rotations, flips, zooming, and shifts.
- Data Split Strategy: 
  * Training Set: 80%
  * Validation Set: 10%
  * Test Set: 10%

--------------------------------------------------------------------------------
METHOD / ML MODEL(S) USED
--------------------------------------------------------------------------------
1. Pre-Screening Leaf Validation Engine:
   - Rule-based texture and color-space pipeline using NumPy and PIL.
   - Color Analysis: RGB-to-HSV conversion evaluating green ratio dominance (Hue 0.2-0.4).
   - Texture Analysis: Grayscale variance computation (`np.var`) to detect leaf surface detail.
   - Edge & Aspect Ratio Analysis: Horizontal and vertical gradient density measurement.

2. Disease Classification Model:
   - Architecture: Deep Convolutional Neural Network (CNN) trained with TensorFlow/Keras.
   - Deployment Optimization: Converted from Keras (`.keras`) format to ONNX (`.onnx`) 
     to drastically reduce server package size from 1.99 GB to ~9.1 MB.
   - Inference Engine: `onnxruntime` C++ execution engine.
   - Hyperparameters (Training):
     * Input Shape: (224, 224, 3)
     * Optimizer: Adam (`learning_rate=0.001`)
     * Loss Function: Categorical Cross-Entropy
     * Activation: ReLU (hidden layers), Softmax (output layer)

--------------------------------------------------------------------------------
EVALUATION AND METRICS
--------------------------------------------------------------------------------
Evaluation Metrics Overview:
- Accuracy: Overall percentage of correctly predicted leaf disease classes.
- Precision: Ratio of true positive predictions to total predicted positives.
- Recall: Ratio of true positive predictions to total actual positives.
- F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: Matrix breakdown of class predictions across 38 categories.

Sample Evaluation Results (Example / Placeholder Table):
+--------------------------------+-----------+-----------+--------+----------+
| Metric                         | Precision | Recall    | F1     | Accuracy |
+--------------------------------+-----------+-----------+--------+----------+
| Leaf Pre-Screening Validation  | 0.8900    | 0.9200    | 0.9046 | 0.9050   |
| 38-Class Disease Model (CNN)   | 0.9780    | 0.9760    | 0.9770 | 0.9780   |
+--------------------------------+-----------+-----------+--------+----------+
Primary Metric: Accuracy (Target >= 95% on test dataset).

--------------------------------------------------------------------------------
RESULTS SUMMARY
--------------------------------------------------------------------------------
The optimized ONNX deep learning pipeline achieves robust performance across 38 
crop disease categories while enforcing input validation to eliminate false positives 
on non-leaf images. Converting the model to ONNX Runtime reduced backend bundle memory 
from 1.99 GB to under 50 MB, enabling sub-second latency on Vercel serverless functions 
without any degradation in prediction accuracy. Future enhancements include 
training a dedicated binary classification model for leaf validation and expanding 
the disease dataset to cover regional crop variations.

--------------------------------------------------------------------------------
REPRODUCIBILITY / ENVIRONMENT
--------------------------------------------------------------------------------
- Required Python Version: Python 3.9, 3.10, or 3.11.
- Virtual Environment Setup (venv):
  ```bash
  python -m venv venv
  # On Windows (PowerShell):
  .\venv\Scripts\Activate.ps1
  # On Linux/macOS:
  source venv/bin/activate
  ```
- Virtual Environment Setup (Conda alternative):
  ```bash
  conda create -n leaf-env python=3.10 -y
  conda activate leaf-env
  ```
- Package Installation:
  ```bash
  pip install -r requirements.txt
  ```

--------------------------------------------------------------------------------
REQUIREMENTS.TXT GUIDANCE
--------------------------------------------------------------------------------
For production deployment (Vercel serverless runtime), use lightweight dependencies:
  fastapi==0.116.1
  uvicorn==0.35.0
  onnxruntime==1.23.2
  pillow==11.3.0
  numpy==2.2.6
  python-multipart==0.0.20

For full offline model retraining, install TensorFlow in your local environment:
  pip install tensorflow tf2onnx

To freeze exact installed dependencies in any environment:
  ```bash
  pip freeze > requirements.txt
  ```

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
1. Convert Keras Model to ONNX (Optional / Maintenance):
   ```bash
   python convert_model.py
   ```
   Output: Generates `final_leaf_disease_model.onnx` (~9.1 MB).

2. Run API Backend Server Locally:
   ```bash
   python api.py
   ```
   Or using Uvicorn directly:
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```
   Expected Output: Server running on `http://localhost:8000`. 
   Documentation available at `http://localhost:8000/docs`.

3. Run Streamlit Testing UI (Local):
   ```bash
   streamlit run app.py
   ```

4. Perform Local Inference via cURL:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@samples/test_image.jpg"
   ```

--------------------------------------------------------------------------------
FILE / DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
leaf_back/
├── api/
│   └── index.py                     # Entry point for Vercel serverless deployment
├── samples/                         # Sample leaf images for manual API testing
├── .gitignore                       # Git ignore rules for virtualenvs and temporary files
├── README.txt                       # Project documentation
├── README_LEAF_DETECTION.md         # Detailed leaf validation pipeline specification
├── api.py                           # Standalone local FastAPI server
├── app.py                           # Streamlit web UI interface
├── convert_model.py                 # Keras to ONNX conversion script
├── final_leaf_disease_model.keras   # Raw trained TensorFlow Keras model
├── final_leaf_disease_model.onnx    # Production lightweight ONNX model
├── main.py                          # Basic FastAPI backend alternative
├── requirements.txt                 # Production dependencies for deployment
└── vercel.json                      # Vercel deployment configuration

--------------------------------------------------------------------------------
CONTACT / ATTRIBUTION
--------------------------------------------------------------------------------
- Author / Maintainer: Bharath Vinayagam
- Project Repository: https://github.com/Bharath-vinayagam/leaf_back

