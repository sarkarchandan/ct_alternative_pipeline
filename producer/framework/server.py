"""
    server.py
    ---------
    Defines routes of the data abstractor process endpoints
"""

import base64
from io import BytesIO
from threading import Thread
import time
from typing import Dict
import json
from kafka import KafkaProducer
from flask import render_template
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from framework import app
from framework.data import DataGenerator


def publish_image_data(generator: DataGenerator) -> None:
    producer: KafkaProducer = KafkaProducer (
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    for idx,sample in enumerate(generator()):
        event: Dict[str, str] = {
            f"sample_{idx}": str(sample.tolist())
        }
        producer.send(topic="image_data", value=event)
        time.sleep(0.5)

generator: DataGenerator = DataGenerator.generator_from(
    config="/home/chandansarkar/workspace/repos/data_pipelines/ct_spark_pipeline/producer/framework/config.yaml")

t: Thread = Thread(target=publish_image_data, args=(generator, ))
t.start()

@app.route("/", methods=["GET"])
def index() -> str:
    """Defines route for the application"""
    fig: Figure = Figure()
    axs: Axes = fig.subplots()
    axs.imshow(generator.base_image)
    buf: BytesIO = BytesIO()
    fig.savefig(buf, format="png")
    img_data: bytes = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template("index.html", img_data=img_data)
    

if __name__ == "__main__":
    pass
