import requests

# Test with real leaf image
with open('test_real_leaf.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    
    print('Real Leaf Test:')
    print(f'Is Leaf: {result["is_leaf"]}')
    print(f'Leaf Confidence: {result["leaf_confidence"]:.2%}')
    print(f'Disease Class: {result["class"]}')
    print(f'Disease Confidence: {result["confidence"]:.2%}')
    print(f'Message: {result["message"]}')
