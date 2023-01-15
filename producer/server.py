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
from pathlib import Path
import sys

from flask import Flask
from kafka import KafkaProducer
from flask import render_template
from matplotlib.figure import Figure
from matplotlib.axes import Axes

framework_path: str = Path().absolute().__str__()
print(f"Abs Path: {framework_path}")
sys.path.append(framework_path)

from framework.gen import DataGenerator

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

# Initializes container process
app: Flask = Flask(__name__)

@app.route("/", methods=["GET"])
def index() -> str:
    """Defines route for the application"""
    fig: Figure = Figure()
    axs: Axes = fig.subplots()
    axs.imshow(generator.abstract_obj_image)
    buf: BytesIO = BytesIO()
    fig.savefig(buf, format="png")
    img_data: bytes = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template("index.html", img_data=img_data)
    

if __name__ == "__main__":
    generator: DataGenerator = DataGenerator.generator_from(
        config="config.yaml")
    t: Thread = Thread(target=publish_image_data, args=(generator, ))
    t.start()
    app.run(host="0.0.0.0", port=9991)
