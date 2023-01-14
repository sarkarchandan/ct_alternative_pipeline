FROM python:3.9.16-bullseye
WORKDIR /producer
COPY . /producer
RUN python -m  pip install -r requirements.txt
CMD [ "python", "main.py" ]