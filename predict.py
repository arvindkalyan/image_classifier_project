import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='./test_images/cautleya_spicata.jpg', type = str)
parser.add_argument('--top_k', default = 5, type = int)
parser.add_argument('--category_names', default = 'label_map.json', type = str)
parser.add_argument('--model', type = str, default = "img_class.h5")

args = parser.parse_args()

img_path = args.img_path
top_k = args.top_k
labels = args.category_names
model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
#print(model.summary())

with open(labels, 'r') as f:
    class_names = json.load(f)
    
def process_image(img):
    image_size = 224
    tensor = tf.convert_to_tensor(img, dtype = tf.int16)
    resize = tf.image.resize(tensor, (image_size,image_size))
    norm = (resize/255)
    return norm.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    preds = model.predict(np.expand_dims(processed_image, axis = 0)).tolist()
    values, labels = tf.math.top_k(preds,top_k)
    return values.numpy().tolist(), labels.numpy().tolist()

probs, labels = predict(img_path, model, top_k)


for i in range(0,top_k):

    print("#" + str(i+1) + " Predicted flower: " + class_names[str(labels[0][i]+1)])    
    print(str(100*probs[0][i])+"%")
