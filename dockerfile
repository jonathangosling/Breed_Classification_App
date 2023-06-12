FROM public.ecr.aws/lambda/python:3.8

CMD mkdir -p /workspace

COPY ./main.py /workspace/main.py
COPY ./requirements.txt /workspace/requirements.txt
COPY ./templates /workspace/templates
COPY ./model.py /workspace/model.py
COPY ./transformations.py /workspace/transformations.py
COPY ./20220815-18301660588220_trained_on_80percent.h5 /workspace/20220815-18301660588220_trained_on_80percent.h5
COPY ./unique_labels.csv /workspace/unique_labels.csv

RUN pip install -r /workspace/requirements.txt
CMD ["/workspace/main.handler"]
