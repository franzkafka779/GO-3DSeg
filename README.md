# GO-3Dseg: 3D Segmentation with Neural Radiance Fields

GO-3Dseg is an advanced 3D segmentation project that combines Neural Radiance Fields (NeRF) with the Segment Anything Model (SAM) and YOLOv8. This project provides an integrated solution for detecting objects in 2D images and performing precise segmentation in 3D space.

## 🚀 Key Features

- **3D Segmentation**: Precise object segmentation in 3D space based on NeRF
- **SAM Integration**: High-quality segmentation using the Segment Anything Model
- **YOLOv8 Support**: 2D object detection and segmentation with YOLOv8-seg model
- **Interactive GUI**: Intuitive interface for real-time 3D segmentation
- **Multi-Dataset Support**: Support for various NeRF datasets including LLFF, LERF, and NeRF Unbounded
- **Real-time Rendering**: High-quality 3D rendering and visualization

## ⚡ Quick Start

### 1. Installation Check
```bash
# Verify installation is correct
python test_installation.py
```

### 2. Basic NeRF Training
```bash
# Train basic NeRF with LLFF dataset
python run.py --config configs/llff/fern.py
```

### 3. 3D Segmentation (GUI Mode)
```bash
# Start interactive 3D segmentation
python run_seg_gui.py --config configs/seg_default.py
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.14+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA 11.6+ support
- **Memory**: Minimum 8GB GPU memory recommended
- **Storage**: At least 10GB free space

### Recommended Specifications
- **GPU**: NVIDIA RTX 3080 or higher
- **Memory**: 16GB GPU memory
- **CPU**: Intel i7 or AMD Ryzen 7 or higher

## 🛠️ Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/your-username/GO-3Dseg.git
cd GO-3Dseg
```

### 2. Environment Setup
```bash
# Create conda environment (recommended)
conda create -n go3dseg python=3.10
conda activate go3dseg

# Install CUDA Toolkit (for GPU usage)
conda install -c anaconda -c conda-forge cudatoolkit=11.6
conda install -c anaconda cudnn
conda install -c conda-forge cudatoolkit-dev
```

### 3. PyTorch Installation
```bash
# Install PyTorch for CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Use automatic installation script (optional)
chmod +x install.sh
./install.sh
```

### 5. Download Model Checkpoints

#### Download SAM Model
```bash
mkdir -p dependencies/sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth
```

#### Download YOLOv8 Model
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
```

### 6. Verify Installation
```bash
# Check if all dependencies are correctly installed
python test_installation.py
```

## 📖 Usage

### Basic NeRF Training

Train NeRF models with various datasets:

```bash
# LLFF datasets
python run.py --config configs/llff/fern.py
python run.py --config configs/llff/flower.py
python run.py --config configs/llff/room.py

# LERF datasets
python run.py --config configs/lerf/figurines.py
python run.py --config configs/lerf/donuts.py

# NeRF Unbounded datasets
python run.py --config configs/nerf_unbounded/bicycle.py
python run.py --config configs/nerf_unbounded/garden.py
```

### 3D Segmentation

#### GUI Mode (Recommended)
```bash
# Start interactive segmentation GUI
python run_seg_gui.py --config configs/seg_default.py

# Segmentation for specific datasets
python run_seg_gui.py --config configs/llff/seg/seg_fern.py
python run_seg_gui.py --config configs/nerf_unbounded/seg_bicycle.py
```

#### Command Line Mode
```bash
# Run segmentation only
python run.py --config configs/seg_default.py --render_segment

# Specific object segmentation
python run.py --config configs/seg_default.py --segment --sp_name "chair"
```

### YOLOv8 Object Detection

```bash
# Single image detection
python yolo_detect.py path/to/your/image.jpg

# Batch processing for multiple images
python yolo_detect.py path/to/image/directory/
```

### Basic Usage Examples

Get started quickly with the included example script:

```bash
# Run basic usage examples
python examples/basic_usage.py
```

## 📁 Project Structure

```
GO-3Dseg/
├── configs/                    # Configuration files
│   ├── default.py             # Default configuration
│   ├── seg_default.py         # Default segmentation config
│   ├── llff/                  # LLFF dataset configurations
│   │   ├── llff_default.py
│   │   ├── fern.py
│   │   ├── flower.py
│   │   └── seg/               # LLFF segmentation configs
│   ├── lerf/                  # LERF dataset configurations
│   │   ├── lerf_default.py
│   │   ├── figurines.py
│   │   └── seg_lerf/          # LERF segmentation configs
│   └── nerf_unbounded/        # NeRF Unbounded configurations
│       ├── nerf_unbounded_default.py
│       ├── bicycle.py
│       └── seg_*.py           # Segmentation configs
├── lib/                       # Core library
│   ├── sam3d.py              # 3D SAM implementation
│   ├── gui.py                # GUI interface
│   ├── render_utils.py       # Rendering utilities
│   ├── load_*.py             # Data loaders
│   ├── seg_*.py              # Segmentation models
│   └── cuda/                 # CUDA kernels
├── dependencies/              # External dependencies
│   └── sam_ckpt/             # SAM model checkpoints
│       └── sam_vit_h_4b8939.pth
├── examples/                  # Usage examples
│   └── basic_usage.py        # Basic usage guide
├── docs/                     # Documentation
│   └── troubleshooting.md    # Troubleshooting guide
├── run.py                    # Main execution file
├── run_seg_gui.py           # Segmentation GUI launcher
├── yolo_detect.py           # YOLOv8 detection script
├── test_installation.py     # Installation test script
├── install.sh               # Automatic installation script
├── requirements.txt         # Python dependencies
└── yolov8x-seg.pt          # YOLOv8 model (after download)
```

## ⚙️ Configuration Options

### Key Configuration Parameters

#### Training Settings
- `N_iters`: Number of optimization iterations (default: 30000)
- `N_rand`: Batch size - number of random rays (default: 1024)
- `lrate_density`: Learning rate for density grid
- `lrate_k0`: Learning rate for feature grid
- `lrate_rgbnet`: Learning rate for RGB network

#### Model Settings
- `num_voxels`: Voxel grid resolution
- `density_type`: Density grid type ('DenseGrid', 'TensoRFGrid')
- `k0_type`: Color/feature grid type
- `rgbnet_depth`: RGB MLP depth
- `rgbnet_width`: RGB MLP width

#### Segmentation Settings
- `segment`: Enable segmentation mode
- `sp_name`: Specify target object name
- `mobile_sam`: Use lightweight SAM model

### Custom Configuration Files

```python
# configs/custom_config.py example
_base_ = './default.py'

# Modify training settings
coarse_train.N_iters = 5000
fine_train.N_iters = 20000

# Modify model settings
coarse_model_and_render.num_voxels = 1024000
fine_model_and_render.num_voxels = 27000000

# Set data path
data.datadir = '/path/to/your/dataset'
data.dataset_type = 'llff'
```

## 🎯 Advanced Usage

### Using Custom Datasets

1. **Prepare LLFF format data**:
```bash
# Calculate camera poses with COLMAP
colmap feature_extractor --database_path database.db --image_path images/
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images/ --output_path sparse/
```

2. **Create configuration file**:
```python
# configs/custom/my_dataset.py
_base_ = '../llff/llff_default.py'

expname = 'my_dataset_experiment'
basedir = './logs'

data = dict(
    datadir='/path/to/my/dataset',
    dataset_type='llff',
    factor=8,  # Image downsampling factor
    llffhold=8,  # Test image interval
)
```

### Batch Processing

Scripts for automatically processing multiple datasets:

```bash
# Train all LLFF datasets
for config in configs/llff/*.py; do
    echo "Training with $config"
    python run.py --config "$config"
done

# Run all segmentation tasks
for config in configs/llff/seg/seg_*.py; do
    echo "Segmenting with $config"
    python run_seg_gui.py --config "$config"
done
```

## 📊 Performance and Benchmarks

### Rendering Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index Measure  
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Segmentation Accuracy Metrics
- **IoU**: Intersection over Union
- **Dice Coefficient**: F1 score for segmentation
- **Pixel Accuracy**: Pixel-level accuracy

### Performance Optimization Tips

1. **GPU Memory Optimization**:
```python
# Start with lower resolution
coarse_model_and_render.num_voxels = 512000
fine_model_and_render.num_voxels = 1280000

# Adjust batch size
coarse_train.N_rand = 512
fine_train.N_rand = 512
```

2. **Training Speed Improvement**:
```python
# Adjust iteration count
coarse_train.N_iters = 2000
fine_train.N_iters = 10000

# Adjust checkpoint frequency
fine_train.i_print = 100
fine_train.i_weights = 2000
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size or resolution
RuntimeError: CUDA out of memory
```
- Reduce `N_rand` value in config (1024 → 512)
- Reduce `num_voxels` value
- Use smaller image resolution (increase `factor` value)

#### 2. SAM Checkpoint Error
```bash
# Solution: Re-download checkpoint
FileNotFoundError: sam_vit_h_4b8939.pth
```
```bash
rm -f dependencies/sam_ckpt/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/sam_ckpt/sam_vit_h_4b8939.pth
```

#### 3. Dependency Conflicts
```bash
# Solution: Reinstall in new environment
conda deactivate
conda env remove -n go3dseg
conda create -n go3dseg python=3.10
conda activate go3dseg
pip install -r requirements.txt
```

#### 4. Data Loading Errors
```bash
# Check the following:
- Verify data path is correct
- Check if image files exist  
- Ensure camera pose file (poses_bounds.npy) exists
```

### Performance Monitoring

Monitor GPU usage and memory:
```bash
# Monitor GPU status
watch -n 1 nvidia-smi

# Check real-time logs
tail -f logs/your_experiment/logs.txt
```

## 🔗 Useful Links

- **Troubleshooting Guide**: [docs/troubleshooting.md](docs/troubleshooting.md)
- **Segment Anything**: https://github.com/facebookresearch/segment-anything
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **NeRF**: https://github.com/bmild/nerf
- **DVGO**: https://github.com/sunset1995/DirectVoxGO

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Environment Setup

```bash
# Install additional development dependencies
pip install -e ".[dev]"

# Code formatting
black lib/ run.py run_seg_gui.py

# Linting
flake8 lib/ run.py run_seg_gui.py

# Type checking
mypy lib/
```

## 📄 License

This project is distributed under the MIT License. See [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgments

- [Segment Anything](https://github.com/facebookresearch/segment-anything) - SAM model
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model  
- [NeRF](https://github.com/bmild/nerf) - Neural Radiance Fields
- [DVGO](https://github.com/sunset1995/DirectVoxGO) - Dense Voxel Grid Optimization
- [LERF](https://github.com/kerrj/lerf) - Language Embedded Radiance Fields

## 📞 Contact and Support

- **Issues**: Please report bugs or request features via GitHub Issues
- **Discussions**: For general questions or discussions, use GitHub Discussions
- **Email**: For direct contact, please reach out via email

---

**GO-3Dseg** - Next-generation 3D Segmentation with Neural Radiance Fields