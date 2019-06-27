# coding:utf-8
from ctypes import *
import cv2
import numpy as np
import os


class Config(Structure):
    _fields_ = [
        ("w", c_int),
        ("h", c_int),
        ("jitter", c_float),
        ("hue", c_float),
        ("saturation", c_float),
        ("exposure", c_float),
    ]


class BBox(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("w", c_float),
        ("h", c_float),
        ("label", c_int)
    ]


class Image(Structure):
    _fields_ = [
        ("w", c_int),
        ("h", c_int),
        ("c", c_int),
        ("bnums", c_int),
        ("data", POINTER(c_float)),
        ("bboxes", POINTER(c_float))
    ]


class DarknetDetectionPreImage:
    def __init__(self, net_params, debug=False):
        os.system('make clean')
        os.system('make')
        self.so_path = './image.so'
        self.lib = CDLL(self.so_path)
        self.image_prehandle = self.lib.image_prehandle
        self.image_prehandle.argtypes = (POINTER(Image), POINTER(Config))
        self.image_prehandle.restype = Image
        self.debug = debug

        self.config = Config()
        self.config.w = net_params.get('w', 1280)
        self.config.h = net_params.get('h', 768)
        self.config.jitter = net_params.get('jitter', 0.3)
        self.config.hue = net_params.get('hue', 0.1)
        self.config.saturation = net_params.get('saturation', 1.5)
        self.config.exposure = net_params.get('exposure', 1.5)

    def process(self, image, bboxes):
        """

        :param image:   image = cv2.imread('../test.jpg')
        :param bboxes:  bboxes = [[2858, 634, 80, 373, 1], [3110, 680, 114, 379, 2]]
        :return:
        """
        image = image.astype(np.float32)
        img = Image()
        img.h, img.w, img.c = image.shape
        img.bnums = len(bboxes)

        if self.debug:
            temp = image.copy()
            for bbox in bboxes:
                x = bbox[0]
                y = bbox[1]
                x2 = x + bbox[2]
                y2 = y + bbox[3]
                cv2.rectangle(temp, (x, y), (x2, y2), (0, 0, 255), thickness=6)
            cv2.imwrite('original.jpg', temp)

        im = np.transpose(image[..., ::-1], (2, 0, 1)).flatten() / 255.0
        img.data = im.ctypes.data_as(POINTER(c_float))
        img.bboxes = np.array(bboxes, dtype=np.float32).ctypes.data_as(POINTER(c_float))
        # enhance image.
        image = self.image_prehandle(byref(img), byref(self.config))
        # get image and bboxes data from C
        result = np.ctypeslib.as_array(image.data, (3, image.h, image.w)) * 255.0
        result = np.transpose(result, (1, 2, 0))[..., ::-1].copy().astype(np.uint8)
        new_bboxes = np.ctypeslib.as_array(image.bboxes, (image.bnums, 5))

        if self.debug:
            for bbox in new_bboxes:
                x = int(bbox[0])
                y = int(bbox[1])
                x2 = int(x + bbox[2])
                y2 = int(y + bbox[3])
                cv2.rectangle(result, (x, y), (x2, y2), (0, 0, 255), thickness=3)
            cv2.imwrite('./result.jpg', result)

        return result, new_bboxes


if __name__ == '__main__':
    img = cv2.imread('../test.jpg')
    bboxes = [[2858, 634, 80, 373, 1],
              [3110, 680, 114, 379, 2]]
    image = DarknetDetectionPreImage({}, debug=True)
    image.process(img, bboxes)










