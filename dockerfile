FROM public.ecr.aws/lambda/python:3.8

WORKDIR /code

COPY ./requirements.txt    /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./main.py     /code/main.py
COPY ./templates /code/templates
COPY ./breeds_ML /code/breeds_ML
COPY ./20220815-18301660588220_trained_on_80percent.h5 /code/20220815-18301660588220_trained_on_80percent.h5
COPY ./unique_labels.csv /code/unique_labels.csv
CMD ["/code/main.handler"]
