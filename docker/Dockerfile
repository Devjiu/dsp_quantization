# Get the base Ubuntu image from Docker Hub
FROM ubuntu:18.04

# Update apps on the base image
#    libopenblas-dev \
#    liblapack-dev \
#    libarpack2-dev \
#    libsuperlu-dev \

# Install the Clang compiler
RUN apt-get update \
  && apt-get install -y ssh \
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    cmake \
    rsync \
    tar \
    python \
    sudo \
    libopenblas-dev \
    liblapack-dev \
    libarpack2-dev \
    libsuperlu-dev \
    git \
  && apt-get clean

# Copy the current folder which contains C++ source code to the Docker image under /usr/src
#COPY ../main.cpp /usr/src/dockertest1
#
RUN service ssh start

RUN useradd -m user && yes password | passwd user
RUN echo " user    ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER user
WORKDIR /home/user

#RUN wget http://sourceforge.net/projects/arma/files/armadillo-9.900.3.tar.xz
#RUN tar xvf armadillo-9.900.3.tar.xz
RUN sudo service ssh start

RUN mkdir third-parties && cd third-parties
RUN git clone https://github.com/xtensor-stack/xtl.git
RUN cd xtl && mkdir build && cd build
RUN cmake .. && make -j 8 && sudo make install
RUN git clone https://github.com/xtensor-stack/xtensor.git
RUN cd xtensor && mkdir build && cd build
RUN cmake .. && make -j 8 && sudo make install
RUN cd ~

CMD ["bash"]
# docker run -it --cap-add sys_ptrace -p127.0.0.1:2222:22 -v C:\Users\Devjiu\CLionProjects\FFT_quantization:/home/user --name clion_remote_
  #env clion/remote-cpp-env:0.