FROM public.ecr.aws/lambda/python:3.8

COPY ./main.py /workspace/main.py
COPY ./requirements.txt /workspace/requirements.txt
COPY ./templates /workspace/templates
COPY ./breeds_ML /workspace/breeds_ML
COPY ./20220815-18301660588220_trained_on_80percent.h5 /workspace/20220815-18301660588220_trained_on_80percent.h5
COPY ./unique_labels.csv /workspace/unique_labels.csv

RUN pip install -r /workspace/requirements.txt
CMD ["/workspace/main.handler"]
