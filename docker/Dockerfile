FROM nvidia/cuda:8.0-cudnn6-devel
MAINTAINER Adam Van Etten

# cd /raid/local/src/yolt2/darknet
# nvidia-docker build -t darknet .
# nvidia-docker run -it -v /raid:/raid --name yolt_train darknet 
# NV_GPU=0,1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
		cmake \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
	 vim \
	 wget \
	 gedit \
	 spyder \
        zlib1g-dev \
        libopencv-dev \
        python-opencv \
        build-essential autoconf libtool libcunit1-dev \
        libproj-dev libgdal-dev libgeos-dev libjson0-dev vim python-gdal \
	dans-gdal-scripts \
	python-tk \
    	# requirements for keras
    	python-h5py \
    	python-yaml \
    	python-pydot \
	graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


RUN pip --no-cache-dir install \
	 #opencv \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
	pandas \
	h5py \
	utm \
	osmnx \
	#easydict \
	scikit-image \
	tensorflow-gpu \
	keras \
        && \
    python -m ipykernel.kernelspec

