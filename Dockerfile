ARG BASE_CONTAINER=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM $BASE_CONTAINER

USER root

COPY requirements.txt .

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# python:3.8.13-slim-buster

#Downloading dependencies 
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc jupyter libpq-dev libsndfile1 ffmpeg cython wget git vim \
&& apt-get -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
# && apt-get -y install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev

# # build kenlm from source
# RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz \
# && mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2

# build kenlm from source
RUN git clone https://github.com/kpu/kenlm.git \
&& mkdir -p kenlm/build \
&& cd kenlm/build \
&& cmake .. \
&& make -j 4

# build ctc-segmentation from source
# RUN git clone https://github.com/lumaku/ctc-segmentation \
# && cd ctc-segmentation \
# && cython -3 ctc_segmentation/ctc_segmentation_dyn.pyx \
# && python setup.py build \
# && python setup.py install --optimize=1 --skip-build

# numpy problems
RUN pip install numpy==1.21.1 --no-binary numpy

# installing dependencies
RUN pip install -r requirements.txt

# installing tensorboardX
RUN pip install tensorboardX --no-cache-dir

# # build ctcdecode from source
# RUN git clone --recursive https://github.com/parlance/ctcdecode.git \
# && cd ctcdecode \ 
# && pip install .

# installing jupyter lab inside
RUN pip install jupyterlab

# declare port used by jupyterlab
EXPOSE 8888

# set default command for jupyterlab
CMD ["jupyter" ,"lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
#CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

WORKDIR /stt_with_kenlm_pipeline
RUN ["bash"]