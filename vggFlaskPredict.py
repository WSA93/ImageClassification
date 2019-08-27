import base64
import io
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app);

def get_model():
    global model
    model = load_model("E:/ML/CNN/vgg.h5")
    print("Model Loaded")

def preprocessImage(image,target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image

print("Loading model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message         = request.get_json(force=True)
    encoded         = message['image']
    decoded         = base64.b64decode(encoded)
    image           = Image.open(io.BytesIO(decoded))
    processed_image = preprocessImage(image,target_size=(224,224))
    
    with graph.as_default():
        prediction = model.predict(processed_image).tolist()
    
    product = "purses"
    confidence = prediction[0][0]

    if(prediction[0][1] > confidence):
        product = "shirts"
        confidence = prediction[0][1]		
    
    if(prediction[0][2] > confidence):
        product = "watches"
        confidence = prediction[0][2]    

    return jsonify(product)        