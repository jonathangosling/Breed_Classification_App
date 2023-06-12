import sys
sys.path.append('/workspace')
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from mangum import Mangum
from tensorflow.image import decode_jpeg, convert_image_dtype, resize
import tensorflow.data as td
from tensorflow.keras.models import load_model
from tensorflow import float32
from tensorflow_hub import KerasLayer
import csv
import numpy as np
# imports for the classifcation model loading, fitting, transforming
# from model import *
# from transformations import *


def load_our_model(model_path):
    """
    Loads model from input path
    """
    # "custom_objects" used to tell Keras that there
    # is an additional (custom) layer (from training)
    model = load_model(model_path,
                       custom_objects={"KerasLayer": KerasLayer})
    return model


def get_unique_labels():
    read_list = []
    with open('/unique_labels.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            read_list = row
    return read_list


# Turn prediction probabilities into the predicted label
def get_pred_label(prediction_probabilites, unique_labels):
    return unique_labels[np.argmax(prediction_probabilites)]

# Define image size
# Ensure image size is of the correct size for the model
# (same as size used in training model)
IMG_SIZE = 224


def process_image(image_content):
    """
    Turns image into a Tensor
    """
    # Turn image into numerical Tensor with 3 colour channels (RGB)
    image = decode_jpeg(image_content, channels=3)
    # Convert colour channel values from 0-255 to 0-1 values
    # (RGB values are 0-255 as standard)
    image = convert_image_dtype(image, float32)
    # Resize the image
    image = resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def create_data_batches(X, y=None, batch_size=1):
    """
    Creates batches of data from image (X) and label (y) pairs.
    Shuffles data if training data but not if validation data.labels.
    Accepts data with no labels (test_data).
    """
    # Create Dataset
    data = td.Dataset.from_tensor_slices(X)
    # Turn Dataset into batches
    # .map applies the image processing function to all images
    # in Dataset .batch creates batches
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch


def get_image_label(image_path, label):
    """
    Creates tuple of (image, label) from input image filepath and label
    """
    image = process_image(image_path)
    return image, label


app = FastAPI()
handler = Mangum(app) # handler for running on AWS Lambda

templates = Jinja2Templates(directory="/workspace/templates")


@app.get('/breed-classifier', response_class=HTMLResponse)
def get_basic_form(request: Request):
    return templates.TemplateResponse("basic_form.html",
                                      {"request": request,
                                       "message": "Submit a .jpg or .png image to get a prediction."})


@app.post('/breed-classifier', response_class=HTMLResponse)
async def get_basic_form_resp(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    if file.filename[-4:] == '.png' or file.filename[-4:] == '.jpg':
        # load the trained image classification model
        model = load_our_model(r'/20220815-18301660588220_trained_on_80percent.h5')
        # transform the image data to the same, batched type as trained
        trans_data = create_data_batches([content])
        # make the prediction
        prediction = model.predict(trans_data)
        # transform the prediction to a predicted species and confidence limit
        unique_labels = get_unique_labels()
        label = get_pred_label(prediction, unique_labels)
        output_message = '''The model predicts that the species of the
         animal in the image that you submitted is {} with a confindence
         of {:2.0f}%.'''.format(label.replace("_", " ").title(),
                                np.max(prediction)*100)
    else:
        output_message = '''Sorry I can't process that,
         the file must be an image of type jpg or png.'''
    return templates.TemplateResponse("basic_form.html",
                                      {"request": request,
                                       "message": output_message})
