from pathlib import Path
import sys
import time
import json
from threading import Thread
from datetime import datetime

import numpy as np
from flask import Flask, render_template, Response
from kafka import KafkaConsumer

framework_path: str = Path().absolute().__str__()
print(f"Abs Path: {framework_path}")
sys.path.append(framework_path)

from framework.recon import Reconstructor

# Initializes container process
app: Flask = Flask(__name__)

@app.route("/", methods=["GET"])
def index() -> str:
    """Defines route for the application"""
    return render_template("index.html", embed="This came from Python")

@app.route("/fetch_status", methods=["GET"])
def fetch_status() -> str:
    return datetime.now().__str__()

def subscribe_for_image_data() -> None:
    consumer: KafkaConsumer = KafkaConsumer(
        'image_data',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my_group_id',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    for events in consumer:
        image: np.ndarray = np.array(list(events.value))
        print(f"Received Image: {image.shape}")
        time.sleep(2)

if __name__ == '__main__':
    t: Thread = Thread(target=subscribe_for_image_data, args=())
    t.start()
    app.run(host="0.0.0.0", port=9992)