import numpy as np
import argparse
import time
from tensorflow import keras
# from tensorflow.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.keras.layers import UpSampling2D,BatchNormalization,Dropout
import cv2
import os
from flask import Flask, request, Response, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
#import binascii

import json
from PIL import Image


#loading data
def load_images(image_): 
    img = cv2.imread(image_,0) 
    img = cv2.resize(img, (540,258))  #width,height      
    img = img/255.0
    img = np.expand_dims(img, axis=-1).astype('float32')
    img = np.expand_dims(img, axis=0).astype('float32')
    return img

    
# Initialize the Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def main():
    # load our input image and grab its spatial dimensions
    #image = cv2.imread("./test1.jpg")
    image = request.files["image"]
    image_name = image.filename
    img = load_images(image_name)
    model = load_model('model.hdf5')
    preds = model.predict(img)
    preds = preds.squeeze()
    preds = preds*255
     # prepare image for response
    _, img_encoded = cv2.imencode('.png', preds)
    response = img_encoded.tostring()
    return Response(response=response, status=200,mimetype="image/png")

    # start flask app
if __name__ == '__main__':
   
    app.run(debug=True,host='0.0.0.0')
