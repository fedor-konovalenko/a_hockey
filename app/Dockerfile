FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Etc
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y python3-pip python3.9-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y cargo rustc
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY Tracking-Anything-with-DEVA /app/Tracking-Anything-with-DEVA
COPY Grounded-Segment-Anything /app/Grounded-Segment-Anything

RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116
RUN pip install -r requirements.txt

ENV AM_I_DOCKER=False
ENV BUILD_WITH_CUDA=True
ENV CUDA_HOME=/usr/local/cuda
ENV TOKENIZERS_PARALLELISM = "false"
ENV CUDA_LAUNCH_BLOCKING = '0'

RUN cd Grounded-Segment-Anything && pip install -e GroundingDINO && pip install -e segment_anything

RUN cd Tracking-Anything-with-DEVA && pip install -e .

CMD ["python3", "src/app.py"]