import sys
sys.path.append('/workspace')
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from mangum import Mangum
from tensorflow.image import decode_jpeg, convert_image_dtype, resize
# imports for the classifcation model loading, fitting, transforming
# from model import *
# from transformations import *

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
        model = load_our_model(r'20220815-18301660588220_trained_on_80percent.h5')
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
