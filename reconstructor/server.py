from pathlib import Path
from typing import List
import base64
import sys
import io
import time
import json
from threading import Thread

from PIL import Image
import numpy as np
from flask import Flask, render_template, Response
from kafka import KafkaConsumer

framework_path: str = Path().absolute().__str__()
print(f"Abs Path: {framework_path}")
sys.path.append(framework_path)

from framework.recon import Reconstructor

# Initializes container process
app: Flask = Flask(__name__)

# Buffer
data_buffer: List[np.ndarray] = []

@app.route("/", methods=["GET"])
def index() -> str:
    """Defines route for the application"""
    return render_template("index.html", embed="This came from Python")

@app.route("/fetch_status", methods=["GET"])
def fetch_status() -> str:
    return str(len(data_buffer))

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
        if events.value == "last_batch":
            break
        deserialized: Image = Image.open(io.BytesIO(
            base64.b64decode(events.value["data"])))
        data_buffer.append(np.array(deserialized))
        time.sleep(2)
    print("Last batch of data received")
    print(f"Length of the buffer: {len(data_buffer)}")
    print(f"After conversion: {np.array(data_buffer).shape}")

if __name__ == '__main__':
    t: Thread = Thread(target=subscribe_for_image_data, args=())
    t.start()
    app.run(host="0.0.0.0", port=9992)
