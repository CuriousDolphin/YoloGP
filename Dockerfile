FROM nvcr.io/nvidia/pytorch:22.10-py3 as torch-img
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as cuda-img



FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base-img
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 ffmpeg  libxext6  libsm6 libgtk2.0-dev pkg-config libgtk2.0-dev  libavcodec-dev python3-opencv libopencv-dev 
RUN apt-get install -y python3-pip
RUN pip3 install torch torchvision  --index-url https://download.pytorch.org/whl/cu118


FROM base-img as app
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY gradio_app.py gradio_app.py
ENTRYPOINT [ "gradio","gradio_app.py" ]

FROM base-img as jupyter-server
WORKDIR /app
RUN pip3 install ipywidgets ipykernel notebook tqdm opencv-python-headless
RUN apt-get install -y libqt5x11extras5
COPY requirements.txt requirements.txt
COPY live_inference.py live_inference.py
COPY helpers.py helpers.py
RUN pip install -r requirements.txt
EXPOSE 8888
RUN jupyter notebook --generate-config
#CMD ["jupyter", "notebook","--allow-root","--ip","0.0.0.0","--NotebookApp.token","f9a3bd4e9f2c3be01cd629154cfb224c2703181e050254b5"] 