FROM public.ecr.aws/lambda/python:3.8
COPY ./requirements.txt    /requirements.txt
COPY ./main.py     /main.py
COPY ./html_files /html_files
COPY ./breeds_ML /breeds_ML
COPY ./20220815-18301660588220_trained_on_80percent.h5 /20220815-18301660588220_trained_on_80percent.h5
COPY ./unique_labels.csv /unique_labels.csv
RUN pip install -r /requirements.txt
CMD ["/main.handler"]
