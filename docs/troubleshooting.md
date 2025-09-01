# SA3D-YOLO 트러블슈팅 가이드

이 문서는 SA3D-YOLO 사용 중 발생할 수 있는 일반적인 문제들과 해결 방법을 제공합니다.

## 🚨 일반적인 문제들

### 1. CUDA 관련 문제

#### 문제: CUDA out of memory
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결 방법:**
- 배치 크기 줄이기: `configs` 파일에서 `N_rand` 값을 줄이세요
- 이미지 해상도 줄이기: `factor` 값을 늘리세요 (예: 4 → 8)
- GPU 메모리 정리: 훈련 전 `torch.cuda.empty_cache()` 실행

#### 문제: CUDA 버전 불일치
```
RuntimeError: CUDA version mismatch
```

**해결 방법:**
```bash
# PyTorch와 CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"
nvidia-smi

# 올바른 PyTorch 버전 설치
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 2. 의존성 설치 문제

#### 문제: torch_scatter 설치 실패
```
ERROR: Could not find a version that satisfies the requirement torch_scatter
```

**해결 방법:**
```bash
# PyTorch 버전에 맞는 torch_scatter 설치
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
```

#### 문제: SAM 모델 로드 실패
```
FileNotFoundError: dependencies/sam_ckpt/sam_vit_h_4b8939.pth
```

**해결 방법:**
```bash
# SAM 체크포인트 다운로드
mkdir -p dependencies/sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth
```

### 3. 데이터셋 관련 문제

#### 문제: 데이터셋 경로 오류
```
FileNotFoundError: [Errno 2] No such file or directory
```

**해결 방법:**
- `configs` 파일에서 `datadir` 경로 확인
- 데이터셋 파일들이 올바른 위치에 있는지 확인
- 상대 경로 대신 절대 경로 사용 고려

#### 문제: 카메라 파라미터 오류
```
ValueError: Invalid camera parameters
```

**해결 방법:**
- 카메라 내부 파라미터 (focal length, principal point) 확인
- 카메라 외부 파라미터 (pose) 형식 확인
- 데이터셋 전처리 스크립트 실행

### 4. GUI 관련 문제

#### 문제: GUI 창이 열리지 않음
```
ImportError: cannot import name 'cv2'
```

**해결 방법:**
```bash
# OpenCV 재설치
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 문제: GUI 반응 없음
```
QApplication: Could not connect to display
```

**해결 방법:**
```bash
# X11 포워딩 설정 (SSH 사용 시)
ssh -X username@server

# 또는 VNC 사용
```

### 5. 성능 관련 문제

#### 문제: 훈련 속도가 느림
- GPU 사용률 확인: `nvidia-smi`
- 배치 크기 증가: `N_rand` 값 늘리기
- 데이터 로딩 최적화: `load2gpu_on_the_fly=True`

#### 문제: 메모리 사용량 과다
- 복셀 그리드 크기 줄이기: `num_voxels` 값 줄이기
- 이미지 해상도 줄이기: `factor` 값 늘리기
- 배치 크기 줄이기: `N_rand` 값 줄이기

## 🔧 디버깅 도구

### 1. 환경 확인 스크립트
```bash
python test_installation.py
```

### 2. GPU 상태 확인
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 실시간 모니터링
```

### 3. 메모리 사용량 확인
```python
import torch
print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 📊 성능 최적화 팁

### 1. GPU 메모리 최적화
```python
# 훈련 루프에서 주기적으로 메모리 정리
if i % 1000 == 0:
    torch.cuda.empty_cache()
```

### 2. 데이터 로딩 최적화
```python
# 데이터를 GPU에 미리 로드
data_dict = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_dict.items()}
```

### 3. 배치 처리 최적화
```python
# 배치 크기를 GPU 메모리에 맞게 조정
N_rand = min(8192, available_gpu_memory // 1024**3 * 1000)
```

## 🐛 알려진 이슈들

### 1. PyTorch 1.12.1 호환성
- 일부 최신 CUDA 버전과 호환성 문제
- 해결책: CUDA 11.6 사용 권장

### 2. SAM 모델 크기
- SAM 체크포인트 파일이 큼 (2.4GB)
- 해결책: SSD 사용 권장

### 3. GUI 반응성
- 대용량 데이터셋에서 GUI 반응 지연
- 해결책: 데이터셋 크기 줄이기

## 📞 지원 요청

문제가 해결되지 않으면 다음 정보와 함께 이슈를 생성해 주세요:

1. **환경 정보:**
   - OS 버전
   - Python 버전
   - PyTorch 버전
   - CUDA 버전
   - GPU 모델

2. **오류 메시지:**
   - 전체 오류 로그
   - 발생 시점
   - 재현 방법

3. **설정 파일:**
   - 사용한 설정 파일
   - 데이터셋 정보

## 🔗 유용한 링크

- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [CUDA 설치 가이드](https://docs.nvidia.com/cuda/)
- [SAM 모델 다운로드](https://github.com/facebookresearch/segment-anything)
- [YOLOv8 문서](https://docs.ultralytics.com/)
