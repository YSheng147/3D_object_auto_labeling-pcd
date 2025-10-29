#!/bin/bash
set -e # 如果任何指令失敗，腳本將會停止

# "=== MS3D 環境安裝腳本 ==="
# "請注意：此腳本假設您在 Ubuntu 22.04 系統上運行"
# "並且具有 sudo 權限來安裝系統套件。"
# "CUDA 11.8 和 CUDNN 8 應該已經由驅動程式安裝。"
# "========================="

# 1. 設定非互動模式 (等同 Dockerfile 中的 debconf)
export DEBIAN_FRONTEND=noninteractive
#sudo echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections

# 2.
echo "--- 正在安裝系統依賴 (需要 sudo) ---"
sudo apt-get update && sudo apt-get install -y \
    wget \
    bzip2 \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential # 添加 build-essential 以確保 gcc/g++ 可用
sudo rm -rf /var/lib/apt/lists/*

echo "--- 正在設定編譯器環境變數 ---"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
echo "CC set to $CC"
echo "CXX set to $CXX"
echo "CUDAHOSTCXX set to $CUDAHOSTCXX"

# 3.
echo "--- 正在安裝 Miniconda ---"
export CONDA_DIR=/opt/conda
export PATH=$CONDA_DIR/bin:$PATH

if [ -d "$CONDA_DIR" ]; then
    echo "Conda 已安裝在 $CONDA_DIR, 跳過安裝。"
else
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    sudo /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
    rm ~/miniconda.sh
    sudo chown -R $USER:$USER $CONDA_DIR
    $CONDA_DIR/bin/conda clean -a -y
fi

echo "--- 正在更新 PATH 以包含 Conda ---"
export PATH=$CONDA_DIR/bin:$PATH

# 4.
echo "--- 正在建立 Conda 環境 ---"
WORKDIR=/docker
cd $WORKDIR

if [ ! -f "ms3d.yaml" ]; then
    echo "錯誤：ms3d.yaml 檔案未在目前目錄 ($WORKDIR) 中找到。"
    exit 1
fi

echo "--- 正在接受 Conda TOS ---"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "--- 正在從 ms3d.yaml 建立環境 ---"
conda env create -f ms3d.yaml

echo "--- 正在初始化 Conda ---"
conda init
echo "--- 將 'conda activate ms3d' 添加到 ~/.bashrc ---"
echo "conda activate ms3d" >> ~/.bashrc

# 5. 進入 Conda 環境安裝 GPU 套件
echo "--- 正在 'ms3d' 環境中安裝 PyTorch 和 GPU 套件 ---"

conda run -n ms3d /bin/bash -c "
set -e
echo '--- 正在安裝 PyTorch, TorchVision, TorchAudio (cu118) ---'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo '--- 正在安裝 torch-scatter ---'
TORCH_VERSION=\$(python -c \"import torch; print(torch.__version__.split('+')[0])\")
echo \"偵測到的 Torch 版本: \$TORCH_VERSION\"
pip install torch-scatter -f https://data.pyg.org/whl/torch-\${TORCH_VERSION}+cu118.html

echo '--- GPU 套件安裝完成 ---'
"

source ~/.bashrc

#pip install torch-scatter --no-binary torch-scatter --no-cache-dir --no-build-isolation