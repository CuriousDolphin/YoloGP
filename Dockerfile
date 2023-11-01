FROM python:3.10 as base-img

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 ffmpeg  libxext6  libsm6 libgtk2.0-dev pkg-config libgtk2.0-dev  libavcodec-dev python3-opencv libopencv-dev 
RUN apt-get install -y python3-pip
RUN pip3 install torch torchvision 

FROM base-img as app
WORKDIR /app
COPY requirements.txt requirements.txt
COPY assets assets
COPY yologp yologp
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python","./yologp/inference_gradio_app.py" ]