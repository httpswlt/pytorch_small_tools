import torch.utils.data as data
from PIL import Image
import os
import os.path
import logging
import numpy as np
import xml.etree.ElementTree as ET


class TVWDarknetDetectionVOC(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    ANNO_EXT = '.xml'
    IMG_EXT_JPG = '.jpg'
    IMG_EXT_JPEG = '.jpeg'

    def convert(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(self, in_file):
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_list = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_list.append([cls_id, bb[0], bb[1], bb[2], bb[3]])
        return out_list

    def __init__(self, image_path, anno_path, train_list_file, class_file, transform=None, target_transform=None):
        self.image_path = image_path
        self.ids = []
        self.transform = transform
        self.target_transform = target_transform
        self.anno_dict = {}
        self.anno_value_dict = {}
        self.img_dict = {}
        if not os.path.exists(train_list_file):
            logging.error("Train file dose not exist!")
            return
        if not os.path.exists(class_file):
            logging.error("Class file dose not exist!")
            return
        with open(train_list_file, 'r') as tlf:
            img_idx = 0
            for f_line in tlf:
                if f_line != "":
                    fline = f_line.strip('\n')
                    self.ids.append(fline)
                    anno_file_path = os.path.join(anno_path, fline + self.ANNO_EXT)
                    if not os.path.exists(anno_file_path):
                        logging.error(anno_file_path + " dose not exist!")
                    self.anno_dict[fline] = anno_file_path
                    img_file_jpg_path = os.path.join(image_path, fline + self.IMG_EXT_JPG)
                    img_file_jpeg_path = os.path.join(image_path, fline + self.IMG_EXT_JPEG)
                    if os.path.exists(img_file_jpeg_path):
                        self.img_dict[fline] = img_file_jpeg_path
                    elif os.path.exists(img_file_jpg_path):
                        self.img_dict[fline] = img_file_jpg_path
                    else:
                        logging.error(fline + " image dose not exist! Only support .jpg and .jpeg")
                img_idx += 1
        self.ids = list(set(self.ids))
        # read class
        self.classes = []
        with open(class_file, 'r') as cf:
            for cline in cf:
                if cline != "":
                    self.classes.append(cline.strip('\n'))

        # read annotation
        for a_id, anno_file in self.anno_dict.items():
            self.anno_value_dict[a_id] = np.array(self.convert_annotation(anno_file))
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.anno_value_dict[img_id]

        img_path = self.img_dict[img_id]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.image_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
