#!/usr/bin/env python3
"""
SA3D-YOLO 기본 사용법 예제
이 스크립트는 SA3D-YOLO의 기본적인 사용법을 보여줍니다.
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from lib.config_loader import Config
from lib import utils
from lib.sam3d import Sam3D
from lib.gui import Sam3dGUI
from lib.bbox_utils import compute_bbox_by_cam_frustrm


def basic_nerf_training():
    """기본 NeRF 훈련 예제"""
    print("🎯 기본 NeRF 훈련 예제")
    print("=" * 50)
    
    # 설정 파일 로드
    config_path = "configs/llff/fern.py"
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일이 없습니다: {config_path}")
        print("   먼저 LLFF 데이터셋을 다운로드하고 설정 파일을 준비해 주세요.")
        return
    
    # 설정 로드
    cfg = Config.fromfile(config_path)
    print(f"✅ 설정 파일 로드: {config_path}")
    
    # 데이터 로드
    data_dict = utils.load_everything(args=None, cfg=cfg)
    print("✅ 데이터 로드 완료")
    
    # 바운딩 박스 계산
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=None, cfg=cfg, **data_dict)
    print(f"📦 바운딩 박스: {xyz_min} ~ {xyz_max}")
    
    print("\n🚀 NeRF 훈련을 시작하려면 다음 명령어를 실행하세요:")
    print(f"   python run.py --config {config_path}")


def basic_segmentation():
    """기본 3D 세그멘테이션 예제"""
    print("\n🎯 기본 3D 세그멘테이션 예제")
    print("=" * 50)
    
    # 설정 파일 확인
    config_path = "configs/seg_default.py"
    if not os.path.exists(config_path):
        print(f"❌ 세그멘테이션 설정 파일이 없습니다: {config_path}")
        return
    
    print("✅ 세그멘테이션 설정 파일 확인")
    
    # SAM 체크포인트 확인
    sam_path = "dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_path):
        print("❌ SAM 체크포인트가 없습니다")
        print("   다음 명령어로 다운로드하세요:")
        print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth")
        return
    
    print("✅ SAM 체크포인트 확인")
    
    print("\n🚀 3D 세그멘테이션을 시작하려면 다음 명령어를 실행하세요:")
    print(f"   python run_seg_gui.py --config {config_path}")


def basic_yolo_detection():
    """기본 YOLOv8 객체 검출 예제"""
    print("\n🎯 기본 YOLOv8 객체 검출 예제")
    print("=" * 50)
    
    # YOLOv8 모델 확인
    yolo_path = "yolov8x-seg.pt"
    if not os.path.exists(yolo_path):
        print("❌ YOLOv8 모델이 없습니다")
        print("   다음 명령어로 다운로드하세요:")
        print("   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt")
        return
    
    print("✅ YOLOv8 모델 확인")
    
    # 테스트 이미지 생성 (간단한 예제용)
    test_image_path = "examples/test_image.jpg"
    if not os.path.exists(test_image_path):
        print("📸 테스트 이미지를 생성합니다...")
        # 간단한 테스트 이미지 생성
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, img)
        print(f"✅ 테스트 이미지 생성: {test_image_path}")
    
    print("\n🚀 YOLOv8 객체 검출을 시작하려면 다음 명령어를 실행하세요:")
    print(f"   python yolo_detect.py {test_image_path}")


def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인")
    print("=" * 50)
    
    # CUDA 확인
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능 (버전: {torch.version.cuda})")
        print(f"   GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA를 사용할 수 없습니다 (CPU 모드)")
    
    # 필수 파일 확인
    required_files = [
        ("configs/default.py", "기본 설정"),
        ("configs/seg_default.py", "세그멘테이션 설정"),
        ("lib/sam3d.py", "SAM 3D 모듈"),
        ("lib/gui.py", "GUI 모듈"),
        ("lib/render_utils.py", "렌더링 유틸리티"),
    ]
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="SA3D-YOLO 기본 사용법 예제")
    parser.add_argument("--check-env", action="store_true", help="환경 설정 확인")
    parser.add_argument("--nerf", action="store_true", help="NeRF 훈련 예제")
    parser.add_argument("--seg", action="store_true", help="세그멘테이션 예제")
    parser.add_argument("--yolo", action="store_true", help="YOLOv8 검출 예제")
    parser.add_argument("--all", action="store_true", help="모든 예제 실행")
    
    args = parser.parse_args()
    
    print("🎉 SA3D-YOLO 기본 사용법 예제")
    print("=" * 60)
    
    if args.check_env or args.all:
        check_environment()
    
    if args.nerf or args.all:
        basic_nerf_training()
    
    if args.seg or args.all:
        basic_segmentation()
    
    if args.yolo or args.all:
        basic_yolo_detection()
    
    if not any([args.check_env, args.nerf, args.seg, args.yolo, args.all]):
        # 기본적으로 모든 예제 실행
        check_environment()
        basic_nerf_training()
        basic_segmentation()
        basic_yolo_detection()
    
    print("\n" + "=" * 60)
    print("📚 더 자세한 정보는 README.md를 참조하세요.")
    print("🔧 문제가 있으면 이슈를 생성해 주세요.")


if __name__ == "__main__":
    main()
