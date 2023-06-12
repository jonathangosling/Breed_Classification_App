# Breed_Classification_App
This is the repo for my FastAPI app using a trained image classification model to provide estimations of the breed of various dogs and cats.

## Contents:
The repo contains:
- main.py: the main python script hosting the FastAPI app. Using Mangum for deployment on AWS Lambda.
- breeds_ML: A directory acting as a python package with all of the necessary functionality to transform an image in the form required by the model and then use the model to make predictions.
- templates: A directory to hold the html files used to return a response to any requests made to the API. This will be accessed using `Jinja2Templates` in the main FastAPI script.
  - basic_form.html: This is the html file which includes a `<form>` element. This allows us to receive an image from the user. When the user submits an image. It is registered as a post request in the app in main.py, activating the `get_basic_form_resp()` function, which uses the model to make a prediction and returns the same html response but with a different 'output message' which is passed a vairable in the `.TemplateResponse` method.
- The trained model in the form of a h5 file.
- unique_labels.csv: csv file containing all of the unique 'labels' i.e. breeds in our model
- Dockerfile, requirments.txt deployment. (Dockerfile used to build docker image, importing dependencies using requirements.txt)
