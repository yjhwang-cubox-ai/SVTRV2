# SVTR Lightning

This project is a PyTorch Lightning implementation of SVTR (Simple Visual Text Recognition) from [MMOCR](https://github.com/open-mmlab/mmocr).
The SVTR model, introduced in ["SVTR: Scene Text Recognition with a Single Visual Model"](https://arxiv.org/abs/2205.00159), provides a simple yet effective approach to text recognition by treating it as a pure vision task.

This implementation offers a more modular and easier-to-use version while maintaining the core functionality of SVTR's text recognition pipeline. It simplifies the original MMOCR implementation by focusing specifically on the SVTR architecture and its training pipeline.

## Overview

The project converts SVTR's recognition pipeline to PyTorch Lightning, offering:
- Data preprocessing and augmentation pipeline
- Learning rate scheduling with warmup and cosine annealing
- Efficient batch processing
- Easy integration with Weights & Biases for experiment tracking

## Installation

```bash
# Clone the repository
git clone git@github.com:yjhwang-cubox-ai/SVTRV2.git
cd mmocr-lightning

# Install dependencies
pip install -r requirements.txt
```

## Usage
# Data Preparation
Prepare your dataset in the following format:
data_dir/
    ├── imgs/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels.json

The labels.json should be formatted as:
Todo

## Acknowledgments
This project is based on MMOCR, an open-source toolbox for text detection, recognition, and understanding. We thank the MMOCR team for their excellent work and open-source contribution.