# .devcontainer/Dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY requirements.txt /tmp/requirements.txt

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC+2
RUN apt-get update && apt-get install -y python3.10-full python3-pip

# Install dependency packpages
RUN python3.10 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r /tmp/requirements.txt
RUN python3.10 -m pip install av2==0.2.1

# Some packages may be useful
RUN python3.10 -m pip install tensorboard
RUN python3.10 -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
RUN python3.10 -m pip install protobuf==3.20.3

RUN apt install -y git curl 
RUN apt-get update  \
    && apt-get install -y libgl1-mesa-dev libglib2.0-0

RUN pip install numpy==1.26.4 opencv-python==4.9.0.80 transformers==4.40.0 torchmetrics==0.7.1 
