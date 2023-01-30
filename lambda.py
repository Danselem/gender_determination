#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

input_size = 224
preprocessor = create_preprocessor('resnet50', target_size=(input_size, input_size))


interpreter = tflite.Interpreter(model_path='eye_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
interpreter.invoke()
output_index = interpreter.get_output_details()[0]['index']


classes = ['female', 'male']

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    score_lite = softmax(float_predictions)

    # return dict(zip(classes, float_predictions))

    return "{}. \n This eye most likely belongs to {} with a {:.2f} percent confidence.".format(
        dict(zip(classes, float_predictions)), classes[np.argmax(score_lite)], 100 * np.max(score_lite))



# def lambda_handler(event, context):
#     url = event['url']
#     result = predict(url)
#     return result


app = Flask('lambda')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'gender_eye/test/Image_3.jpg'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)
