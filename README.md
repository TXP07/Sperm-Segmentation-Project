# Sperm-Segmentation-Project

## Quick Tour<br>
This is a pytorch implementation for our MICCAI-2023 paper "A Multi-Task Method for Immunofixation Electrophoresis Image Classification".<br>
To run the code To run the code, please make sure you have prepared your data following the same structure as follows (you can also refer to the examplar data in this repository):<br>
If you wish to utilize any of their notebooks or data, you can easily copy the required files from [segment-anything](https://github.com/facebookresearch/segment-anything) to your local copy of this repository.<br>
## Project Structure<br>
Here is an overview of the project's directory structure:<br>
```
Sperm-Segmentation-Project/
    - Sperm.ipynb                 # Main script for model training
    - Sperm.py            # Main script for model training
    - segment-anything-main/       #SAM model
        - sam_vit_b_01ec64.pth 
        - sam_vit_h_4b8939.pth
        - sam_vit_l_0b3195.pth
    - README.md             # This file
    - requirements.txt      # Lists the required Python packages
    - train.py              # Main script for model training
    - data/                 # Directory for storing your dataset
        - original_images     # Original images
        - Preprocessing_images      # Preprocessing images

```
## Data Organization<br>
To structure your data for training and testing, it is recommended to organize your dataset as follows:<br>
```
- data/                 # Directory for storing your dataset
        - original_images     # Original images
            - 000.jpg
            - 001.jpg
            - ...
            - N.jpg
        - Preprocessing_images      # Preprocessing images
```
Here's how the data is organized:<br>

- 'original_images' contains original images you want to process.
- 'Preprocessing_images' contains preprocessing images and they will be given automatically by the code and it will be empty before running.<br>


You can add data to it yourself, and the resolution of the image specification is 720*540. Or you can directly use the sample data we provide.<br>
Example of how images and masks looked like in my dataset:<br>
## Getting Started<br>
Follow these steps to set up and start working with the project:<br>

1. **Environment Setup**: Create a virtual environment and install the required dependencies.Steps omitted here.<br>
2. **SAM model Setup**:The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.<br>

Install Segment Anything:

```
cd segment-anything; pip install -e .
```

  The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
3. **Run the code**： Please use jupyter notebook to open the .ipynb file and run it, or use software such as pycharm to run the .py file.Choose the serial number of image you like and modify it in the code.And you can run the code now.<br>
## Example Result:<br>
## References：<br>
















