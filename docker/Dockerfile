# 1. 基礎映像檔 
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# 設定非互動模式
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# 2. 安裝系統依賴 (wget, bzip2 用於安裝 conda)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDAHOSTCXX=/usr/bin/g++

# 3. 安裝 Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -a -y
    
# 4. 建立 Conda 環境 
WORKDIR /app
COPY ms3d.yaml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f ms3d.yaml

RUN conda init bash
RUN echo "conda activate ms3d" >> /root/.bashrc

# 進入conda安裝gpu套件
SHELL ["conda", "run", "-n", "ms3d", "/bin/bash", "-c"]

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu118.html
