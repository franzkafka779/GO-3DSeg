#!/bin/bash

# SA3D-YOLO 설치 스크립트
# 이 스크립트는 Ubuntu/Debian 기반 시스템에서 SA3D-YOLO를 설치합니다.

set -e

echo "🚀 SA3D-YOLO 설치를 시작합니다..."

# 시스템 업데이트
echo "📦 시스템 패키지를 업데이트합니다..."
sudo apt-get update

# 필수 시스템 패키지 설치
echo "🔧 필수 시스템 패키지를 설치합니다..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Conda 설치 (없는 경우)
if ! command -v conda &> /dev/null; then
    echo "🐍 Conda를 설치합니다..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Conda 환경 생성
echo "🔨 Conda 환경을 생성합니다..."
conda create -n sa3d-yolo python=3.10 -y
conda activate sa3d-yolo

# CUDA Toolkit 설치
echo "⚡ CUDA Toolkit을 설치합니다..."
conda install -c anaconda -c conda-forge cudatoolkit=11.6 -y
conda install -c anaconda cudnn -y
conda install -c conda-forge cudatoolkit-dev -y

# PyTorch 설치
echo "🔥 PyTorch를 설치합니다..."
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Python 의존성 설치
echo "📚 Python 의존성을 설치합니다..."
pip install -r requirements.txt

# SAM 체크포인트 다운로드 (없는 경우)
if [ ! -f "dependencies/sam_ckpt/sam_vit_h_4b8939.pth" ]; then
    echo "📥 SAM 체크포인트를 다운로드합니다..."
    mkdir -p dependencies/sam_ckpt
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth
fi

# YOLOv8 모델 다운로드 (없는 경우)
if [ ! -f "yolov8x-seg.pt" ]; then
    echo "📥 YOLOv8 모델을 다운로드합니다..."
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
fi

echo "✅ 설치가 완료되었습니다!"
echo ""
echo "🎉 다음 명령어로 환경을 활성화하고 사용할 수 있습니다:"
echo "conda activate sa3d-yolo"
echo ""
echo "🚀 사용 예시:"
echo "python run.py --config configs/llff/fern.py"
echo "python run_seg_gui.py --config configs/seg_default.py"
echo "python yolo_detect.py path/to/your/image.jpg"
