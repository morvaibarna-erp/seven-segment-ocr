import numpy as np
import tensorflow as tf
import cv2
import pathlib

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

for file in pathlib.Path('test').iterdir():

    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = cv2.resize(img, (30, 200))

    interpreter.set_tensor(input_details[0]['index'], [new_img])

    interpreter.invoke()
    rects = interpreter.get_tensor(
        output_details[0]['index'])
    scores = interpreter.get_tensor(
        output_details[2]['index'])
    
    print("For file {}".format(file.stem))
    print("Rectangles are: {}".format(rects))
    print("Scores are: {}".format(scores))

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)