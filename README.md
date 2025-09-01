# SA3D-YOLO: 3D Segmentation with Neural Radiance Fields

SA3D-YOLO는 Neural Radiance Fields (NeRF)와 YOLOv8을 결합한 3D 세그멘테이션 프로젝트입니다. 이 프로젝트는 2D 이미지에서 객체를 검출하고 3D 공간에서 세그멘테이션을 수행할 수 있는 통합 솔루션을 제공합니다.

## 🚀 주요 기능

- **3D 세그멘테이션**: NeRF 기반 3D 공간에서의 객체 세그멘테이션
- **YOLOv8 통합**: YOLOv8-seg 모델을 활용한 2D 객체 검출 및 세그멘테이션
- **대화형 GUI**: 실시간 3D 세그멘테이션을 위한 대화형 인터페이스
- **다양한 데이터셋 지원**: LLFF, LERF, NeRF Unbounded 등 다양한 NeRF 데이터셋 지원
- **실시간 렌더링**: 고품질 3D 렌더링 및 시각화

## 📋 요구사항

### 시스템 요구사항
- Python 3.8+
- CUDA 11.6+ (GPU 사용 시)
- 최소 8GB GPU 메모리 권장

### 주요 의존성
- PyTorch 1.12.1+
- CUDA Toolkit 11.6
- OpenCV
- Ultralytics (YOLOv8)
- Segment Anything Model (SAM)

## 🛠️ 설치

1. **저장소 클론**
```bash
git clone https://github.com/your-username/SA3D-YOLO.git
cd SA3D-YOLO
```

2. **환경 설정**
```bash
# Conda 환경 생성 (권장)
conda create -n sa3d-yolo python=3.10
conda activate sa3d-yolo

# CUDA Toolkit 설치
conda install -c anaconda -c conda-forge cudatoolkit=11.6
conda install -c anaconda cudnn
conda install -c conda-forge cudatoolkit-dev

# PyTorch 설치
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **SAM 체크포인트 다운로드**
```bash
# SAM 모델 체크포인트가 dependencies/sam_ckpt/ 디렉토리에 있는지 확인
# sam_vit_h_4b8939.pth 파일이 필요합니다
```

## 🚀 사용법

### 1. 기본 NeRF 훈련
```bash
python run.py --config configs/llff/fern.py
```

### 2. 3D 세그멘테이션 (GUI 모드)
```bash
python run_seg_gui.py --config configs/seg_default.py
```

### 3. YOLOv8 객체 검출
```bash
python yolo_detect.py path/to/your/image.jpg
```

### 4. 설정 파일 커스터마이징
각 데이터셋별로 설정 파일을 수정하여 모델 파라미터를 조정할 수 있습니다:
- `configs/llff/`: LLFF 데이터셋 설정
- `configs/lerf/`: LERF 데이터셋 설정  
- `configs/nerf_unbounded/`: NeRF Unbounded 데이터셋 설정

## 📁 프로젝트 구조

```
SA3D-YOLO/
├── configs/                 # 설정 파일들
│   ├── llff/               # LLFF 데이터셋 설정
│   ├── lerf/               # LERF 데이터셋 설정
│   ├── nerf_unbounded/     # NeRF Unbounded 설정
│   └── seg_default.py      # 세그멘테이션 기본 설정
├── lib/                    # 핵심 라이브러리
│   ├── sam3d.py           # 3D SAM 구현
│   ├── gui.py             # GUI 인터페이스
│   ├── render_utils.py    # 렌더링 유틸리티
│   └── ...
├── dependencies/           # 외부 의존성
│   └── sam_ckpt/          # SAM 모델 체크포인트
├── run.py                 # 메인 실행 파일
├── run_seg_gui.py         # 세그멘테이션 GUI 실행
├── yolo_detect.py         # YOLOv8 검출 스크립트
└── requirements.txt       # Python 의존성
```

## 🎯 주요 컴포넌트

### 1. NeRF 기반 3D 렌더링
- Dense Voxel Grid 기반 NeRF 구현
- 고품질 3D 렌더링 및 시각화
- 다양한 데이터셋 포맷 지원

### 2. SAM 3D 세그멘테이션
- Segment Anything Model의 3D 확장
- 대화형 3D 세그멘테이션
- 실시간 마스크 생성 및 편집

### 3. YOLOv8 통합
- YOLOv8-seg 모델을 활용한 2D 객체 검출
- 2D-3D 세그멘테이션 연동
- 자동 객체 인식 및 분류

## 🔧 설정 옵션

### 훈련 설정
- `N_iters`: 최적화 반복 횟수
- `N_rand`: 배치 크기 (랜덤 레이 수)
- `lrate_*`: 각 컴포넌트별 학습률
- `weight_*`: 손실 함수 가중치

### 모델 설정
- `num_voxels`: 복셀 그리드 크기
- `density_type`: 밀도 그리드 타입
- `k0_type`: 색상/특성 그리드 타입
- `rgbnet_*`: 색상 MLP 설정

## 📊 성능 및 벤치마크

- **렌더링 품질**: PSNR, SSIM, LPIPS 메트릭 지원
- **세그멘테이션 정확도**: IoU, Dice coefficient 측정
- **실시간 성능**: GPU 메모리 최적화

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [Segment Anything](https://github.com/facebookresearch/segment-anything) - SAM 모델
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 객체 검출 모델
- [NeRF](https://github.com/bmild/nerf) - Neural Radiance Fields
- [DVGO](https://github.com/sunset1995/DirectVoxGO) - Dense Voxel Grid Optimization

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**SA3D-YOLO** - 3D 세그멘테이션의 새로운 패러다임
