from flask import Flask, request, redirect, render_template, url_for
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np


app = Flask(__name__)

@app.route('/', methods = ['GET','POST']) # The route decorator is used to bind a function to a URL
def index():
    if request.method=='POST':
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = load_model('keras_Model.h5', compile=False)

        # Load the labels
        class_names = open('labels.txt', 'r').readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        pimage= request.form['filename']        
        # Replace this with the path to your image
        image = Image.open('images/test/' + pimage).convert('RGB')

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array
        try:
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print('Class:', class_name, end='')
            print('Confidence score:', confidence_score)
           
            #return redirect('/')
            return render_template('layout.html', value=class_name, x=confidence_score*100, y=pimage)
        
            
        except:
            print("couldnt classify image")
    else:
        return render_template('index.html')
       
if __name__ == "__main__":
    app.run(host='127.0.0.1',debug=True)
