FROM python:3.9.16-bullseye
ENV FLASK_APP="/data_abstractor/framework/__init__.py"
WORKDIR /data_abstractor
COPY . /data_abstractor
RUN python -m  pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "main.py" ]