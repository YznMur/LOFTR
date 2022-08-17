ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"


FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# RUN pip install pytorch-lightning numpy Pillow scikit-image opencv-contrib-python albumentations timm 

RUN pip install wandb matplotlib



RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
COPY ./requirements.txt /home/loftr/

RUN pip install --upgrade pip
RUN pip install torch einops yacs kornia


RUN pip install -r /home/loftr/requirements.txt

ARG USER=trainer
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user 

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN cd /home/${USER} && chown -R ${UID}:${GID} ./

USER ${UID}:${GID}