import json
from kafka import KafkaProducer


if __name__ == "__main__":
    producer: KafkaProducer = KafkaProducer(
        value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        bootstrap_servers=["kafka:9092"]
    )
    producer.send(
        topic="example_data", 
        value= {
            "rot_int": 5,
            "angle_start": 0,
            "angle_end": 180,
            "img_dim": 100,
            "pad_len": 10,
        }
    )
