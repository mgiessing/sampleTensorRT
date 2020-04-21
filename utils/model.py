# import os
# import sys
# import tarfile
# import requests
# import tensorrt as trt
# from utils.paths import PATHS



class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (3, 416, 416)
    # Name of output node
    OUTPUT_NAME = "NMS"

    OUTPUT_NAME = "conv_reg"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]

