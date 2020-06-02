# sampleTensorRT

## Requirements:
- Python 3.6/3.7
- TensorRT >= v5.x
- PyCuda (which requires Cython)
    - Numpy (should be installed through pycuda)
    - opencv (should be installed through pycuda)
- A trained model with the following files:
    - .prototxt file
    - .caffemodel file
    - .json file (for FRCNN)
    - anchors.txt file (for yolov2)
    - labelfile
    - Imagefile for single testing or camerastream

## Usage:
### Two scenarios: 
1. You want to use graphical output a display attached to a device (e.g. Jetson Nano/TX2 etc.) or running X11 forwarding:
 - Use `python3 detector_deploy.py --help` to see all possible (and required) arguments + optional ones.
2. You just want to check the logic in terminal without graphical output:
 - Just run the script with the required arguments as seen under the 'Execution' heading below and set `showgui=False`.<br>

See YoloV2 for concrete instructions

## Notes:
 - INT8 quantization doesn't work yet (the engine building fails), this might get fixed in a new release, therefore choose FP16 or FP32 precision at the moment
