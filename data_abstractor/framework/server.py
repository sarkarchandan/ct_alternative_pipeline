"""
    server.py
    ---------
    Defines routes of the data abstractor process endpoints
"""

import json
from flask import request
from flask import render_template
from kafka import KafkaProducer

from framework import app

@app.route("/", methods=["POST", "GET"])
def index() -> str:
    """Defines route for the application"""
    if request.method == "POST":
        payload={
                "rot_int": int(request.form.get("rot_int")),
                "angle_start": int(request.form.get("angle_start")), 
                "angle_end":int(request.form.get("angle_end")),
                "img_dim":int(request.form.get("img_dim")), 
                "pad_len":int(request.form.get("padding"))
            }
        producer: KafkaProducer = KafkaProducer(
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
        bootstrap_servers=["kafka:9092"]
        )
        producer.send(topic="Topic", value=payload)
        return render_template("index.html")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    pass
