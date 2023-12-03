import os
import time

from flask import Flask, request, render_template, send_file
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

app = Flask(__name__)
denoising_model = load_model('denoising_autoencoder.h5')
model = load_model('cnn_model.h5')


def get_class_name(label_number):
    # Flat list of all class names
    str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                  'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                  'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                  'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                  'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
                  'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
                  'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
                  'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                  'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                  'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                  'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # Use the label number to look up the class name
    try:
        class_name = str_labels[label_number]
    except IndexError:
        class_name = 'Unknown label number'

    return class_name


@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')


@app.route('/classification', methods=['GET', 'POST'])
def index():
    label_name = None  # initialize label_name as None
    filename = None  # initialize filename as None
    cache_buster = int(time.time())
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = "uploaded_image.jpg"
            image_path = f"static/{filename}"  # Save to static directory
            uploaded_file.save(image_path)
            img = image.load_img(image_path, target_size=(32, 32))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            predictions = model.predict(x)
            label = np.argmax(predictions[0])
            label_number = int(label)
            label_name = get_class_name(label_number)
    return render_template('classification.html', label_name=label_name, filename=filename, cache_buster=cache_buster)


@app.route('/autoencoder', methods=['GET', 'POST'])
def denoise():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            original_image_path = os.path.join('static', 'original_image.jpg')
            uploaded_file.save(original_image_path)
            img = image.load_img(original_image_path, color_mode="grayscale", target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            denoised_image = denoising_model.predict(x)
            denoised_image_path = os.path.join('static', 'denoised_image.jpg')
            image.save_img(denoised_image_path, denoised_image[0] * 255.0, color_mode="grayscale")
            cache_buster = int(time.time())
            return render_template('autoencoder.html',
                                   original_filename='original_image.jpg',
                                   denoised_filename='denoised_image.jpg',
                                   cache_buster=cache_buster)
    return render_template('autoencoder.html', original_filename=None, denoised_filename=None)



if __name__ == '__main__':
    app.run(debug=True)
