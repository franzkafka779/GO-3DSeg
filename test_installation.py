#!/usr/bin/env python3
"""
SA3D-YOLO 설치 테스트 스크립트
이 스크립트는 모든 의존성이 올바르게 설치되었는지 확인합니다.
"""

import sys
import importlib
import os
import torch
import cv2
import numpy as np

def test_import(module_name, package_name=None):
    """모듈 import 테스트"""
    try:
        if package_name is None:
            package_name = module_name
        module = importlib.import_module(module_name)
        print(f"✅ {package_name} import 성공")
        return True
    except ImportError as e:
        print(f"❌ {package_name} import 실패: {e}")
        return False

def test_cuda():
    """CUDA 사용 가능 여부 테스트"""
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능 (버전: {torch.version.cuda})")
        print(f"   사용 가능한 GPU: {torch.cuda.device_count()}개")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("⚠️  CUDA를 사용할 수 없습니다 (CPU 모드로 실행됩니다)")
        return False

def test_sam_checkpoint():
    """SAM 체크포인트 파일 존재 여부 테스트"""
    sam_path = "dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
    if os.path.exists(sam_path):
        file_size = os.path.getsize(sam_path) / (1024 * 1024 * 1024)  # GB
        print(f"✅ SAM 체크포인트 파일 존재 ({file_size:.2f} GB)")
        return True
    else:
        print("❌ SAM 체크포인트 파일이 없습니다")
        print("   다음 명령어로 다운로드하세요:")
        print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth")
        return False

def test_yolo_model():
    """YOLOv8 모델 파일 존재 여부 테스트"""
    yolo_path = "yolov8x-seg.pt"
    if os.path.exists(yolo_path):
        file_size = os.path.getsize(yolo_path) / (1024 * 1024 * 1024)  # GB
        print(f"✅ YOLOv8 모델 파일 존재 ({file_size:.2f} GB)")
        return True
    else:
        print("❌ YOLOv8 모델 파일이 없습니다")
        print("   다음 명령어로 다운로드하세요:")
        print("   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt")
        return False

def test_config_files():
    """설정 파일 존재 여부 테스트"""
    config_paths = [
        "configs/default.py",
        "configs/seg_default.py",
        "configs/llff/llff_default.py",
        "configs/lerf/lerf_default.py",
        "configs/nerf_unbounded/nerf_unbounded_default.py"
    ]
    
    all_exist = True
    for path in config_paths:
        if os.path.exists(path):
            print(f"✅ {path} 존재")
        else:
            print(f"❌ {path} 없음")
            all_exist = False
    
    return all_exist

def test_lib_modules():
    """lib 모듈들 존재 여부 테스트"""
    lib_modules = [
        "lib.sam3d",
        "lib.gui", 
        "lib.render_utils",
        "lib.load_data",
        "lib.utils"
    ]
    
    all_exist = True
    for module in lib_modules:
        if test_import(module, module):
            pass
        else:
            all_exist = False
    
    return all_exist

def main():
    """메인 테스트 함수"""
    print("🧪 SA3D-YOLO 설치 테스트를 시작합니다...\n")
    
    # Python 버전 확인
    print(f"🐍 Python 버전: {sys.version}")
    
    # 기본 의존성 테스트
    print("\n📦 기본 의존성 테스트:")
    basic_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("imageio", "ImageIO"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("tqdm", "TQDM"),
    ]
    
    basic_success = True
    for module, name in basic_deps:
        if not test_import(module, name):
            basic_success = False
    
    # 고급 의존성 테스트
    print("\n🔬 고급 의존성 테스트:")
    advanced_deps = [
        ("ultralytics", "Ultralytics (YOLOv8)"),
        ("segment_anything", "Segment Anything Model"),
        ("transformers", "Transformers"),
        ("lpips", "LPIPS"),
        ("supervision", "Supervision"),
    ]
    
    advanced_success = True
    for module, name in advanced_deps:
        if not test_import(module, name):
            advanced_success = False
    
    # CUDA 테스트
    print("\n⚡ CUDA 테스트:")
    cuda_available = test_cuda()
    
    # 파일 존재 여부 테스트
    print("\n📁 파일 존재 여부 테스트:")
    sam_exists = test_sam_checkpoint()
    yolo_exists = test_yolo_model()
    configs_exist = test_config_files()
    
    # lib 모듈 테스트
    print("\n🔧 lib 모듈 테스트:")
    lib_modules_exist = test_lib_modules()
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 테스트 결과 요약:")
    print("="*50)
    
    if basic_success:
        print("✅ 기본 의존성: 모든 패키지가 정상적으로 설치되었습니다")
    else:
        print("❌ 기본 의존성: 일부 패키지 설치에 문제가 있습니다")
    
    if advanced_success:
        print("✅ 고급 의존성: 모든 패키지가 정상적으로 설치되었습니다")
    else:
        print("❌ 고급 의존성: 일부 패키지 설치에 문제가 있습니다")
    
    if cuda_available:
        print("✅ CUDA: GPU 가속이 가능합니다")
    else:
        print("⚠️  CUDA: CPU 모드로만 실행 가능합니다")
    
    if sam_exists:
        print("✅ SAM 체크포인트: 파일이 존재합니다")
    else:
        print("❌ SAM 체크포인트: 파일이 없습니다")
    
    if yolo_exists:
        print("✅ YOLOv8 모델: 파일이 존재합니다")
    else:
        print("❌ YOLOv8 모델: 파일이 없습니다")
    
    if configs_exist:
        print("✅ 설정 파일: 모든 설정 파일이 존재합니다")
    else:
        print("❌ 설정 파일: 일부 설정 파일이 없습니다")
    
    if lib_modules_exist:
        print("✅ lib 모듈: 모든 모듈이 정상적으로 로드됩니다")
    else:
        print("❌ lib 모듈: 일부 모듈 로드에 문제가 있습니다")
    
    # 전체 성공 여부
    all_success = (basic_success and advanced_success and sam_exists and 
                   yolo_exists and configs_exist and lib_modules_exist)
    
    print("\n" + "="*50)
    if all_success:
        print("🎉 모든 테스트가 통과했습니다! SA3D-YOLO를 사용할 준비가 완료되었습니다.")
        print("\n🚀 사용 예시:")
        print("   python run.py --config configs/llff/fern.py")
        print("   python run_seg_gui.py --config configs/seg_default.py")
        print("   python yolo_detect.py path/to/your/image.jpg")
    else:
        print("⚠️  일부 테스트가 실패했습니다. 위의 오류 메시지를 확인하고 문제를 해결해 주세요.")
        print("\n💡 해결 방법:")
        print("   1. requirements.txt의 모든 패키지를 설치했는지 확인")
        print("   2. SAM 체크포인트와 YOLOv8 모델을 다운로드했는지 확인")
        print("   3. CUDA와 PyTorch 버전이 호환되는지 확인")
    
    print("="*50)

if __name__ == "__main__":
    main()
