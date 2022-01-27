from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from pathlib import Path



# Import fast.ai Library
import fastai
from fastai.vision.all import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# to solve "NotImplementedError: cannot instantiate 'PosixPath' on your system"
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

print('Loading our Inference model...')
learn = load_learner('export.pkl')
print('Model Loaded')



def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
    pred_class,pred_idx,outputs = learn.predict(img_path)
    return pred_class
    




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    
    app.run(debug=True)


