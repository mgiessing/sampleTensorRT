# Copyright 2019. IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#encoding=utf8
'''
Detection with Tiny YoloV2
In this example, we will load a Tiny YoloV2 model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
import cv2
from datetime import datetime
import tensorrt as trt

import utils.inference as inference_utils # TRT inference wrappers
import utils.model as model_utils #
import utils.postprocessing as post_utils
from utils.paths import PATHS # Path management
import time

WINDOW_NAME = 'Camera'

def load_anchors(filepath):
    with open(filepath) as f:
        anchors = [l.strip() for l in f]
        anchors = np.array([i.split(",") for i in anchors], dtype=np.float32)
        anchors = np.reshape(anchors, (2, int(anchors.shape[1]/2)), order='F')
        return anchors

def load_classes(filepath):
    with open(filepath) as f:
        labels = [l.strip() for l in f]
        lbl = np.asarray([i.split() for i in labels])
        d = {} 
        for val in lbl:
            d[val[1]] = val[0]
        lst_sorted = sorted(d.items())
        lbls = []
        for i in range(len(lst_sorted)):
            lbls.append(lst_sorted[i][1])
        return lbls

def draw_box(img, name, box, score):
    """ draw a single bounding box on the image """
    xmin, ymin, xmax, ymax = box

    box_tag = '{} : {:.2f}'.format(name[:2], score)
    text_x, text_y = 5, 7

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (xmin, ymin-boxsize[1]-text_y),
                (xmin+boxsize[0]+text_x, ymin), (0, 225, 0), -1)
    cv2.putText(img, box_tag, (xmin+text_x, ymin-text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def show_results(img, results, i):
    """ draw bounding boxes on the image """
    img_width, img_height = img.shape[1], img.shape[0]
    disp_console = True
    imshow = True
    for result in results:
        box_x, box_y, box_w, box_h = [int(v) for v in result[1:5]]
        if disp_console:
            print('    class : {}, [x,y,w,h]=[{:d},{:d},{:d},{:d}], Confidence = {}'.\
                format(result[0], box_x, box_y, box_w, box_h, str(result[5])))
        xmin, xmax = max(box_x-box_w//2, 0), min(box_x+box_w//2, img_width)
        ymin, ymax = max(box_y-box_h//2, 0), min(box_y+box_h//2, img_height)

        if imshow:
            draw_box(img, result[0], (xmin, ymin, xmax, ymax), result[5])
    if imshow:
        cv2.imshow('YOLO detection', img)

class Detector():
    def __init__(self, proto, model, labelmap_file, gpu_mode, trt_mode, batch_size, resolution, precision):

        self.batch_size = batch_size

        TRT_PRECISION_TO_DATATYPE = {
            16: trt.DataType.HALF,
            32: trt.DataType.FLOAT
        }
        trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[precision]

        tyolo_model_path = model
        tyolo_deploy_path = proto

        trt_engine_path = PATHS.get_engine_path(trt_engine_datatype, batch_size)
        print("trt_engine_path:", trt_engine_path)
        try:
            os.makedirs(os.path.dirname(trt_engine_path))
        except:
            pass

        # override native resolution with command line or prototxt
        try:
            import re
            if resolution != "":
                wh = resolution.split('x')
                w = int(wh[0])
                h = int(wh[1])

                f = open(tyolo_deploy_path, 'r')
                contents = f.read()
                f.close()
                contents = re.sub('(?<=input: "data"\ninput_shape {\n  dim: 1\n  dim: 3\n  dim: )[0-9]+', str(h), contents)
                contents = re.sub('(?<=input: "data"\ninput_shape {\n  dim: 1\n  dim: 3\n  dim: ' + str(h) + '\n  dim: )[0-9]+', str(w), contents)
                f = open(tyolo_deploy_path, 'w')
                f.write(contents)
                f.close()
            else:
                f = open(tyolo_deploy_path, 'r')
                contents = f.read()
                f.close()
                dim = re.search('(?<=input: "data"\ninput_shape {\n  dim: 1\n  dim: 3\n  dim: )[0-9]+', contents)
                h = contents[dim.span()[0]:dim.span()[1]]
                dim = re.search('(?<=input: "data"\ninput_shape {\n  dim: 1\n  dim: 3\n  dim: ' + h + '\n  dim: )[0-9]+', contents)
                w = int(contents[dim.span()[0]:dim.span()[1]])
                h = int(h)

            import utils.model as model_utils
            model_utils.ModelData.INPUT_SHAPE = (3, h, w)
        except Exception as e:
            import logging
            logging.info('Bad resolution, using defaults')

        # Set up all TensorRT data structures needed for inference
        self.trt_inference_wrapper = inference_utils.TRTInference(
            tyolo_deploy_path, trt_engine_path, tyolo_model_path,
            trt_engine_datatype=trt_engine_datatype,
            batch_size=batch_size)

    def detect(self, image, conf_thresh=0.5, classes=None, anchors=None):
        '''
        Tiny YoloV2 detection
        '''
        # Start measuring time
        loadimage_start_time = time.time()

        # handle post files, get folders and comma separated file paths/names
        from os import listdir
        from os.path import isdir, isfile, join
        if isinstance(image, list):
            images = []
            for im in image:
                if isdir(im):
                    images.extend([join(im, f) for f in listdir(im) if isfile(join(im, f))])
                else:
                    images.append(im)
        elif isinstance(image, str):
            images = image.split(',')
        elif isinstance(image, np.ndarray):
            images = [image]

        # actual number of images
        actual_batch_size = len(images)

        image_w = [0] * actual_batch_size
        image_h = [0] * actual_batch_size

        # images2 = np.zeros((actual_batch_size, 3, model_utils.ModelData.INPUT_SHAPE[1], model_utils.ModelData.INPUT_SHAPE[2]), np.float32)
        transformed_images = np.zeros((actual_batch_size, 3, model_utils.ModelData.INPUT_SHAPE[1], model_utils.ModelData.INPUT_SHAPE[2]), np.float32)
        for i, image in enumerate(images):
            if isinstance(image, str):
                image = cv2.imread(image)
            images[i] = image
            image_w[i], image_h[i] = image.shape[1], image.shape[0]
            image = cv2.resize(image, (model_utils.ModelData.INPUT_SHAPE[2], model_utils.ModelData.INPUT_SHAPE[1]), interpolation=cv2.INTER_AREA)
            image = image / 255.
            image = np.asarray(image).transpose([2,0,1]).astype(np.float32)
            #image = cv2.subtract(image, (104.0, 117.0, 123.0, 0.0))
            transformed_images[i] = image

        print("Image loading time: {} ms for {} images".format(int(round((time.time() - loadimage_start_time) * 1000)), actual_batch_size))
        t_start = datetime.now()
        # Get TensorRT Tiny YoloV2 model output
        detection_out = self.trt_inference_wrapper.infer_batch(transformed_images)
        total = datetime.now()-t_start
        print('Total time for inference was {:.2f} milliseconds'.format((total).total_seconds()*1e3))
        h_output = detection_out.reshape((len(transformed_images), -1, 13, 13))
        for i in range(len(h_output)):
            print("Output of frame #{}:".format(i+1))
            results = post_utils.get_candidate_objects(h_output[i], images[i].shape, classes, anchors, conf_thresh)
            show_results(images[i], results, i)
        #print("Average inference time was:", (total.total_seconds()*1e3)/(i+1), "milliseconds, or:", 1/((total.total_seconds())/(i+1)), "FPS")

def main(argv):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Caffe-YOLO detection test')
    parser.add_argument('--proto', required = True, type=str, help='.prototxt file')
    parser.add_argument('--model', required = True, type=str, help='.caffemodel file')
    parser.add_argument('--anchors', required = True, type=str, help='anchors.txt file')
    parser.add_argument('--labels', required = True, type=str, help='label.txt file')
    parser.add_argument('--image', required = True, type=str, help='input image')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--resolution', default="416x416", type=str, help='image resolution')
    parser.add_argument('--precision', default=16, type=int, choices=[8, 16, 32], help='precision mode (8: INT8, 16: HALF or 32: FLOAT)')
    parser.add_argument('--confthre', default = 0.5, type=float, help = "Override confidence threshold")

    def open_window(width, height):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, width, height)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

    open_window(1280, 720)

    args, _ = parser.parse_known_args()

    detector = Detector(args.proto, args.model, args.labels, True, True, args.bs, args.resolution, args.precision)
    if args.image != 'Camera':
        detector.detect(args.image, args.confthre, load_classes(args.labels), load_anchors(args.anchors))
    else:
        gst_str = ('nvarguscamerasrc ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
        #cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture('/dev/video0')
        if cap.isOpened():
            while True:
                _, img = cap.read()
                detector.detect(img, args.confthre, load_classes(args.labels), load_anchors(args.anchors))
                key = cv2.waitKey(1)
                if key == 27:  # ESC key: quit program
                    break
                if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                    break
            cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
