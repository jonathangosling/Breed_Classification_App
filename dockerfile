FROM public.ecr.aws/lambda/python:3.8

WORKDIR /

COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
RUN pip install -r /code/requirements.txt --target  "${LAMBDA_TASK_ROOT}"

COPY templates .
COPY model.py ${LAMBDA_TASK_ROOT}
COPY transformations.py ${LAMBDA_TASK_ROOT}
COPY 20220815-18301660588220_trained_on_80percent.h5 .
COPY unique_labels.csv .
CMD ["/main.handler"]
