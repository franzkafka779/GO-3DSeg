from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sa3d-yolo",
    version="0.1.0",
    author="SA3D-YOLO Contributors",
    author_email="your-email@example.com",
    description="3D Segmentation with Neural Radiance Fields and YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/SA3D-YOLO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "sa3d-yolo=run:main",
            "sa3d-seg=run_seg_gui:main",
            "yolo-detect=yolo_detect:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.py", "*.yaml", "*.yml", "*.json"],
    },
)
