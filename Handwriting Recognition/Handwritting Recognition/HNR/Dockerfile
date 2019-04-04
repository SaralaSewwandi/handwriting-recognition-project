
FROM ubuntu:18.04


ENV DEBIAN_FRONTEND noninteractive
# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
	#	libopenjpeg2 \
#		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
#		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
	       #libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
##		python-dev \
		python-tk \
		python-numpy \
		python3-dev \
		python3-tk \
		python3-numpy \
		ant \
		default-jdk \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* 
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
#	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-nose \
		python3-h5py \
		python3-skimage \
		python3-matplotlib \
		python3-pandas \
		python3-sklearn \
		python3-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN 	apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y --no-install-recommends python python-dev python-pip build-essential cmake git pkg-config libjpeg8-dev libtiff5-dev  libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev gfortran libavresample-dev libgphoto2-dev libgstreamer-plugins-base1.0-dev libdc1394-22-dev  && \
	pip install numpy && \
	cd /opt && \
	git clone https://github.com/opencv/opencv_contrib.git && \
	cd opencv_contrib && \
	git checkout 3.4.0 && \	
	cd /opt && \
	git clone https://github.com/opencv/opencv.git && \
	cd opencv && \
	git checkout 3.4.0 && \
	mkdir build && \
	cd build && \
	cmake 	-D CMAKE_BUILD_TYPE=RELEASE \
		-D BUILD_NEW_PYTHON_SUPPORT=ON \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D INSTALL_C_EXAMPLES=OFF \
		-D INSTALL_PYTHON_EXAMPLES=OFF \
		-D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
		-D PYTHON_EXECUTABLE=/usr/bin/python2.7 \
		-D BUILD_EXAMPLES=OFF /opt/opencv && \
	make -j $(nproc) && \
	make install && \
	ldconfig && \
	apt-get purge -y git && \
	apt-get clean && rm -rf /var/lib/apt/lists/* && \
	rm -rf /opt/opencv*


RUN apt-get update && apt-get install -y \
	libghc-edit-distance-dev && \
	python3.6  
#        python3-pip

#	libpng12-dev 

RUN apt-get update && apt-get install -y \
	python3-pip && \
	python3-pip install plotly \
	python3-pip install numpy \
        python3-pip install scipy \
        python3-pip install nose \
       	python3-pip install h5py \
       	python3-pip install skimage \
        python3-pip install matplotlib \
        python3-pip install pandas \
        python3-pip install sklearn \
        python3-pip install sympy 

#jupitor notebook installation
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyter


