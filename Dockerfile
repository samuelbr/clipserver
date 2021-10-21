FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY app.py app.py
COPY config.yaml config.yaml

CMD [ "python3", "app.py" ]