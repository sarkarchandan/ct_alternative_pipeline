FROM wurstmeister/kafka:latest

ENV KAFKA_BROKER_ID=1
ENV KAFKA_LISTENERS=PLAINTEXT://:9092
ENV KAFKA_ADVERTISED_HOST_NAME=localhost
ENV KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
ENV ALLOW_PLAINTEXT_LISTENER=yes
ENV KAFKA_CREATE_TOPICS=example_data:1:1:compact
ENV KAFKA_AUTO_CREATE_TOPICS_ENABLE=true
ENV KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181

# RUN apt update \
#     && apt install -y gnupg apt-transport-https ca-certificates curl software-properties-common \
#     && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt update \
#     && apt-get install -y python3 python3-dev \
#     && python3.9 -m pip install kafka

EXPOSE 9092
