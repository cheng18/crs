FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /crs_elmo

RUN apt-get update && apt-get install vim -y
RUN apt-get update && apt-get install git -y
RUN pip install --upgrade pip
RUN pip install tensorflow-hub

# OpenCC
RUN apt-get update && apt-get install python make cmake gcc g++ doxygen -y
RUN cd / && git clone https://github.com/BYVoid/OpenCC.git
RUN cd /OpenCC && make && make install
RUN pip install OpenCC