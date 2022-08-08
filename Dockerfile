FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git
RUN pip3 install --upgrade pip
RUN mkdir /app
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 4002
ENTRYPOINT ["python", "run.py"]

