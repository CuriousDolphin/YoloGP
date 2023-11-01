FROM python:3.10 as base-img

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 ffmpeg  libxext6  libsm6 libgtk2.0-dev pkg-config libgtk2.0-dev  libavcodec-dev python3-opencv libopencv-dev 
RUN apt-get install -y python3-pip
RUN pip3 install torch torchvision 

FROM base-img as app

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
COPY requirements.txt requirements.txt
COPY assets assets
COPY yologp yologp
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python","yologp/inference_gradio_app.py" ]