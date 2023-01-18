# pylint: disable=wrong-import-position
"""
    server.py
    ---------
    Defines utilities for the reconstructor process
"""

from pathlib import Path
from typing import List, Any, Tuple
import base64
import sys
import io
import time
import json
import concurrent.futures
from threading import Thread
from PIL import Image
import numpy as np
from flask import Flask, render_template, request
from kafka import KafkaConsumer

framework_path: str = Path().absolute().__str__()
sys.path.append(framework_path)

from framework.recon import Reconstructor, Reconstruction
from framework import utils
from framework import vis

# Initializes container process
app: Flask = Flask(__name__)

# Buffer for incoming data
data_buffer: List[np.ndarray] = []

# Declaration of the reconstructor object, which would be instantiated 
reconstructor: Reconstructor


def is_last_batch() -> bool:
    """Convenient function to detect the last batch of events"""
    cfg: utils.Config = utils.Config()
    ds_len: int = np.arange(
        start=int(cfg[utils.KEY_START_ANGlE]),
        stop=int(cfg[utils.KEY_END_ANGLE]),
        step=int(cfg[utils.KEY_ROTATION_INTERVAL])).shape[0]
    return len(data_buffer) == ds_len


@app.route("/", methods=["GET", "POST"])
def index() -> Any:
    """Defines route for the application"""
    if request.method == "GET":
        return render_template("index.html", ftc_st="Ready To Receive")
    elif request.method == "POST":
        cfg: utils.Config = utils.Config()
        rot_prof: Tuple[int, int, int] = (
            int(cfg[utils.KEY_START_ANGlE]),
            int(cfg[utils.KEY_END_ANGLE]),
            int(cfg[utils.KEY_ROTATION_INTERVAL])
        )
        global reconstructor
        reconstructor = Reconstructor(
            dataset=data_buffer,
            rotation_profile=rot_prof
        )
        sgs: np.ndarray = reconstructor.create_sinogram().transpose()
        sinogram: bytes = vis.serialize(
            source=sgs,
            pcolor=True,
            lsp=reconstructor.linear_space,
            angles=reconstructor.angles_rad,
            x_label=r"$\theta$ (rad)",
            y_label=r"$\ln({I_0}/I)$",
            title="Sinogram")
        recon_result: np.ndarray
        with concurrent.futures.ThreadPoolExecutor() as exec:
            future: Any = exec.submit(
                reconstructor.reconstruct,
                Reconstruction.FILTERED_BACK_PROJECTION)
            recon_result = future.result()
        reconstruction: bytearray = vis.serialize(
            source=recon_result,
            title="Reconstructed with Filtered Back Projection")
        return render_template(
            "index.html",
            sinogram=sinogram,
            reconstruction=reconstruction)


@app.route("/fetch_status", methods=["GET"])
def fetch_status() -> str:
    if is_last_batch():
        return "last_batch"
    return str(len(data_buffer))


def subscribe_for_image_data() -> None:
    consumer: KafkaConsumer = KafkaConsumer(
        "image_data",
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="my_group_id",
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    for events in consumer:
        if events.value == "last_batch":
            break
        deserialized: Image = Image.open(io.BytesIO(
            base64.b64decode(events.value["data"])))
        data_buffer.append(np.array(deserialized))
        time.sleep(1)
    print("Last batch of data received")
    print(f"Length of the buffer: {len(data_buffer)}")


if __name__ == '__main__':
    t: Thread = Thread(target=subscribe_for_image_data, args=())
    t.start()
    app.run(host="0.0.0.0", port=9992)
