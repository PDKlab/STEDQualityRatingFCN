FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         curl \
         bzip2 \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy scipy mkl pyyaml typing && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN conda install -c pytorch cuda91 pytorch=0.3.1=py36_cuda9.1.85_cudnn7.0.5_2 && conda clean -ya
RUN conda install scikit-image matplotlib flask && conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY . /workspace/executable
