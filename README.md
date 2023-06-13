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
- Dockerfile, requirments.txt, buildspec.yaml for deployment.
  
## Notes:
- Things can get a little tricky when importing custom packages in AWS lambda. It appears that the current working directory when the lambda function executes the application file (`main.py`) is not necessarily the same directory as the application file itself, meaning it is unable to find the custom packages.
  - I think there could be a number of solutions to this. Here's what I did:
    1. Create a new directory (`workspace`) in the dockerfile and COPY all files into this directory.
    2. In the application file (`main.py`) append the directory (`sys.path.append('/workspace')`). Note: we added all of our files to this new directory so that we can just append that, rather than appending the route (`/`) which contains everything, limiting the amount that the interpreter has to search.
  - Another solution could be to change the current working directory from the python script using `os.chdir()`
  - I also found [this blog post](https://xebia.com/blog/python-and-relative-imports-in-aws-lambda-functions/) discussing using relative imports for python in AWS Lambda functions, where they take yet a different approach.
  - I haven't properly tested this yet, but checking [the AWS developer guide post](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html). I can see a potential solution which may be better:
    - The AWS base images (i.e. `public.ecr.aws/lambda/python:3.8` in this case) actually provides environment variables `LAMBDA_TASK_ROOT` and `LAMBDA_RUNTIME_DIR` which are assigned values pointing to task and runtime directories. The AWS documentation actually suggests installing dependencies to the directory found using the variable `LAMBDA_TASK_ROOT` "alongside the function handler to ensure that the Lambda runtime can locate them when the function is invoked."
    - In the dockerfile:
       1. `RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"`
       2. `COPY app.py ${LAMBDA_TASK_ROOT}`
       3. Also copy all other (custom) dependencies to ${LAMBDA_TASK_ROOT}
- The default memory and timeout configurations on AWS lambda can cause issues when importing large packages like tensorflow. This can be solved by:
  - First: make sure that you are only importing the necessary functionality (`from tensorflow. ... import ...`).
  - Second: increase the timeout in the AWS lambda configuration to an reasonable amount.
  - Finally: if it's still taking too long, you can increase the speed of imports by increasing the CPU provisioned. This can be altered in the lambda configuration by increasing the memory allocated ("Your function is allocated CPU proportional to the memory configured").