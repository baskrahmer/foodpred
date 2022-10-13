FROM public.ecr.aws/lambda/python:3.9

RUN yum install gcc -y

COPY requirements.txt ./
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r requirements.txt -t .

COPY app/model ./model
RUN chmod -R 755 ./model

COPY app/embeddings.npy app/raw.npy app/lci_data.yaml ./
COPY app/config.yaml ./

COPY app/app.py app/app_helpers.py ./

CMD ["app.lambda_handler"]
