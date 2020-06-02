# sampleTinyYoloV2 TensorRT

## Requirements:
- Python 3.6/3.7
- TensorRT >= v5.x
- PyCuda (which requires Cython)
    - Numpy (should be installed through pycuda)
    - opencv (should be installed through pycuda)
- A trained tiny yolo v2 model with the following files:
    - .prototxt file
    - .caffemodel file
    - anchors.txt file
    - label.txt file
    - Imagefile for single testing or camerastream

## Usage:
### Two scenarios: 
1. You want to use graphical output a display attached to a device (e.g. Jetson Nano/TX2 etc.) or running X11 forwarding:
 - Use `python3 detector_deploy.py --help` to see all possible (and required) arguments + optional ones.
2. You just want to check the logic in terminal without graphical output:
 - Just run the script with the required arguments as seen under the 'Execution' heading below.<br>

### Execution:
 For my scenario I have trained a model which detects 5 different types/colours of light-emitting diodes (LEDs) and created another folder `testdata` with all the required files inside this repository. I used the second scenario and commented the mentioned lines which would create/destroy the window out.
 
 ```
 python3 detector_deploy.py --proto testdata/deploy.prototxt --model testdata/model.caffemodel --anchors testdata/anchors.txt --label testdata/labels.txt --image testdata/testimage.jpg
[...]
Output of frame #1:
    class : blue_led, [x,y,w,h]=[243,185,27,32], Confidence = 1.0
    class : blue_led, [x,y,w,h]=[214,114,31,27], Confidence = 1.0
    class : yellow_led, [x,y,w,h]=[237,106,31,32], Confidence = 1.0
    class : red_led, [x,y,w,h]=[262,297,37,24], Confidence = 0.9999994
    class : yellow_led, [x,y,w,h]=[178,205,22,32], Confidence = 0.9999989
    class : yellow_led, [x,y,w,h]=[313,223,41,24], Confidence = 0.99999833
    class : blue_led, [x,y,w,h]=[317,176,22,36], Confidence = 0.9999975
    class : green_led, [x,y,w,h]=[176,180,36,24], Confidence = 0.9999968
    class : red_led, [x,y,w,h]=[119,72,32,33], Confidence = 0.99999547
    class : white_led, [x,y,w,h]=[85,258,36,28], Confidence = 0.99999356
    class : green_led, [x,y,w,h]=[229,201,31,31], Confidence = 0.99999046
    class : red_led, [x,y,w,h]=[262,130,38,21], Confidence = 0.99999034
    class : white_led, [x,y,w,h]=[165,42,35,28], Confidence = 0.99998987
    class : red_led, [x,y,w,h]=[274,163,25,35], Confidence = 0.99998105
    class : green_led, [x,y,w,h]=[197,243,29,28], Confidence = 0.9999808
    class : white_led, [x,y,w,h]=[80,155,35,31], Confidence = 0.9998838
 ```
Multiple images can be passed comma seperated, for a continous camerastream pass the `--image='Camera'` instead of a filepath. Currently the camerastream will force `--showgui = True`, but in a later release this might be optional.

## Notes:
 - INT8 quantization doesn't work yet (the engine building fails), this might get fixed in a new release, therefore choose FP16 or FP32 precision at the moment
