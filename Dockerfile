FROM public.ecr.aws/lambda/python:3.9

RUN yum install gcc -y

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install torch==2.0.0 -f https://download.pytorch.org/whl/cpu/torch

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY model/model.py harrygobert/model/model.py

COPY model ./model
RUN chmod -R 755 ./model

COPY raw.npy lci_data.yaml ./
COPY config.yaml ./

COPY app.py app_helpers.py ./

CMD ["app.lambda_handler"]