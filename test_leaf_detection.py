import requests
import json
import os
from PIL import Image
import numpy as np

def test_leaf_detection(image_path, api_url="http://localhost:8000"):
    """
    Test the leaf detection API with a given image
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        # Test leaf detection only
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect-leaf", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"\n🔍 Testing: {image_path}")
            print(f"📊 Is Leaf: {result['is_leaf']}")
            print(f"🎯 Confidence: {result['confidence']:.2%}")
            print(f"📝 Message: {result['message']}")
            
            if 'basic_detection' in result:
                print(f"   Basic Detection: {result['basic_detection']['is_leaf']} ({result['basic_detection']['confidence']:.2%})")
                print(f"   Advanced Detection: {result['advanced_detection']['is_leaf']} ({result['advanced_detection']['confidence']:.2%})")
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")

def test_disease_prediction(image_path, api_url="http://localhost:8000"):
    """
    Test the full disease prediction API with a given image
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/predict", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"\n🌿 Testing Disease Prediction: {image_path}")
            print(f"📊 Is Leaf: {result['is_leaf']}")
            print(f"🎯 Leaf Confidence: {result['leaf_confidence']:.2%}")
            
            if result['is_leaf']:
                print(f"🦠 Disease Class: {result['class']}")
                print(f"🎯 Disease Confidence: {result['confidence']:.2%}")
                print(f"📝 Message: {result['message']}")
            else:
                print(f"❌ Not a leaf: {result['message']}")
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")

def create_test_images():
    """
    Create some test images to demonstrate the system
    """
    print("🖼️  Creating test images...")
    
    # Create a simple green rectangle (simulating a leaf)
    leaf_img = Image.new('RGB', (224, 224), color='green')
    leaf_img.save('test_leaf.jpg')
    print("✅ Created test_leaf.jpg (green rectangle)")
    
    # Create a blue rectangle (non-leaf)
    non_leaf_img = Image.new('RGB', (224, 224), color='blue')
    non_leaf_img.save('test_non_leaf.jpg')
    print("✅ Created test_non_leaf.jpg (blue rectangle)")
    
    # Create a complex pattern (non-leaf)
    pattern = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pattern_img = Image.fromarray(pattern)
    pattern_img.save('test_pattern.jpg')
    print("✅ Created test_pattern.jpg (random pattern)")

if __name__ == "__main__":
    print("🧪 Leaf Disease Detection Test Suite")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/docs")
        print("✅ API is running")
    except:
        print("❌ API is not running. Please start the API first with: python api.py")
        print("\n🖼️  Creating test images for later use...")
        create_test_images()
        exit(1)
    
    # Create test images if they don't exist
    if not all(os.path.exists(f) for f in ['test_leaf.jpg', 'test_non_leaf.jpg', 'test_pattern.jpg']):
        create_test_images()
    
    print("\n🧪 Testing Leaf Detection Only...")
    test_leaf_detection('test_leaf.jpg')
    test_leaf_detection('test_non_leaf.jpg')
    test_leaf_detection('test_pattern.jpg')
    
    print("\n🧪 Testing Full Disease Prediction...")
    test_disease_prediction('test_leaf.jpg')
    test_disease_prediction('test_non_leaf.jpg')
    test_disease_prediction('test_pattern.jpg')
    
    print("\n🎯 Test completed!")
    print("\n💡 Tips:")
    print("- Upload real leaf images to test disease classification")
    print("- Upload non-leaf images to test rejection")
    print("- The system now rejects non-leaf images before classification")
    print("- You can use /detect-leaf endpoint for leaf-only detection")
    print("- Use /predict endpoint for full disease classification")
