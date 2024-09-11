from flask import Flask, request, jsonify
import base64
import os
import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import os.path
import sys

app = Flask(__name__)

# Directory where images will be saved
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return "Image Upload API is running"

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get JSON data from the request
    data = request.get_json()
    
    if not data or 'image' not in data or 'name' not in data:
        return jsonify({"error": "Image data or name not provided"}), 400
    
    image_base64 = data['image']
    image_name = data['name']
    
    # Decode the base64 string
    try:
        image_data = base64.b64decode(image_base64)
    except Exception as e:
        return jsonify({"error": "Invalid base64 string"}), 400
    
    # Save the image with the provided name
    image_path = os.path.join(UPLOAD_FOLDER, image_name)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    result = predict(image_path, "model_float16.tflite")
    text = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])
    print(f'Extracted text: {text}')
    
    return jsonify({"message": "Image uploaded successfully", "path": image_path, "text":text}), 200


alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

def prepare_input(image_path):
  input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  input_data = cv2.resize(input_data, (200, 31))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data

def predict(image_path, model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_data = prepare_input(image_path)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output = interpreter.get_tensor(output_details[0]['index'])
  return output


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')

