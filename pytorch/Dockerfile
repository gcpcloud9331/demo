FROM python:3.9.6-buster
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir pytorch-pretrained-bert Flask transformers
CMD [ "python", "app.py" ]
