"""
    server.py
    ---------
    Defines routes of the data abstractor process endpoints
"""

import base64
from io import BytesIO
from typing import Dict, List
from threading import Thread
import time
import json
from pathlib import Path
import sys

import cv2
from flask import Flask
from kafka import KafkaProducer
from flask import render_template, request
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

framework_path: str = Path().absolute().__str__()
sys.path.append(framework_path)

from framework.gen import DataGenerator
from framework import vis

def publish_image_data(generator: DataGenerator) -> None:
    """Publishes abstract 2D object sample data set to the predetermined 
    kafka topic

    Args:
        generator: Generator object for abstract data set
    """
    producer: KafkaProducer = KafkaProducer (
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    for sample in generator():
        serialized: str = base64.b64encode(
            cv2.imencode(".png", sample)[1]).decode()
        payload: Dict[str, str] = {
            "data": serialized,
        }
        producer.send(topic="image_data", value=payload)
        time.sleep(0.5)
    producer.send(topic="image_data", value="last_batch")

# Initializes container process
app: Flask = Flask(__name__)

# Initialize DataGenerator
generator: DataGenerator = DataGenerator.from_config()

@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Defines route for the application"""
    num_images: int = 5
    vis_images: List[np.ndarray] = list(generator())[:num_images]
    img_data: bytes = vis.serialize(images=vis_images, rows=1, cols=num_images)
    if request.method == "POST":
        if request.form.get("stream_button") == "stream":
            print("Producer: Publishing asynchronously...")
            t: Thread = Thread(target=publish_image_data, args=(generator, ))
            t.start()
            return render_template("index.html", 
                img_data=img_data, 
                publish_state="Streaming Data")
    elif request.method == "GET":
        return render_template("index.html", img_data=img_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9991)
