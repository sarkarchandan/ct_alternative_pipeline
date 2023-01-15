import time
import json
from kafka import KafkaConsumer

if __name__ == '__main__':
    consumer: KafkaConsumer = KafkaConsumer(
        'image_data',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my_group_id',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    for events in consumer:
        event_data = events.value
        print(f'Received: {event_data}')
        time.sleep(2)