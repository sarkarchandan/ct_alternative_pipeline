import json
from kafka import KafkaProducer


if __name__ == "__main__":
    producer: KafkaProducer = KafkaProducer(
        value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        bootstrap_servers=["kafka:9092"]
    )
    producer.send(topic="Topic", value={"hello": "producer"})
