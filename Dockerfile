FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas matplotlib seaborn sklearn

WORKDIR /app

COPY "Boston_prediction.py" /app

CMD ["python3", "-u", "./Boston_prediction.py"]