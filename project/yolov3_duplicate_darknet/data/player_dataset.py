# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import cv2
import torch
import random
import math


class YoloDataSets(Dataset):
    debI = 0

    def __init__(self, data_path, input_size=(1280, 768), batch_size=4, image_set='trainval', jitter_x=0.3,
                 jitter_y=0.3, augment=True, hue=0.1, saturation=1.5, exposure=1.5, angle=0):
        """

        :param data_path: dataset path(should contain Annotations, JPEGImages and so on.)
        :param image:
        :param target_transform:
        :param pre_process:
        """
        self.input_size_width = input_size[0]
        self.input_size_height = input_size[1]
        self.data_path = data_path
        #self.data_set_path = os.path.join(self.data_path, 'ImageSets', 'Main', image_set + '.txt')
        self.data_set_path = image_set
        self._anno_path = os.path.join(self.data_path, 'Annotations')
        self._img_path = os.path.join(self.data_path, 'JPEGImages')
        self.batch_count = 0
        self.augment = augment
        with open(self.data_set_path, 'r') as f:
            img_files = f.read().splitlines()
            img_files = list(filter(lambda x: len(x) > 0, img_files))
            self.img_files = [os.path.join(self._img_path, ix+'.jpg') for ix in img_files]
        n = len(self.img_files)
        self.ids = img_files
        self.jitter_x = jitter_x
        self.jitter_y = jitter_y
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.label_files = [x.replace('JPEGImages', 'labels').
                                replace('.jpeg', '.txt').
                                replace('.jpg', '.txt').
                                replace('.bmp', '.txt').
                                replace('.png', '.txt') for x in self.img_files]

        self.labels = [np.zeros((0, 5))] * n
        self.labels_pre = [np.zeros((0, 5))] * n


        iter = tqdm(self.label_files, desc='Reading labels') if n > 1000 else self.label_files
        for i, file in enumerate(iter):
            try:
                with open(file, 'r') as f:
                    self.labels_pre[i] = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                with open(file, 'r') as f:
                    self.labels[i] = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                self.labels[i][:, :4] = self.labels_pre[i][:, 1:5]
                self.labels[i][:, 4] = self.labels_pre[i][:, 0]

            except:
                print("Missing " + file)
                pass  # missing label file


    @staticmethod
    def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      borderValue=(127.5, 127.5, 127.5)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        if targets is None:
            targets = []
        border = 0  # width of added border (optional)
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets

    @staticmethod
    def constrain(min, max, value):
        if value > max:
            return max
        if value < min:
            return min
        return value

    @staticmethod
    def rand():
        return int(random.random() * 32767)

    def __getitem__(self, index):
        img_id = self.img_files[index]
        img = None
        img = cv2.imread(img_id)
        target = self.labels[index]

        img_height, img_width, img_channel = img.shape

        # for bbox in target:
        #     print(bbox)  # x y w h
        #     left = bbox[1] - bbox[3] / 2.
        #     right = left + bbox[3]
        #     top = bbox[2] - bbox[4] / 2.
        #     bottom = top + bbox[4]
        #
        #     bbx_lt = (int((bbox[1] - bbox[3] / 2.) * self.input_size_width),
        #               int((bbox[2] - bbox[4] / 2.) * self.input_size_height))
        #     bbx_rb = (int((bbox[1] + bbox[3] / 2.) * self.input_size_width),
        #               int((bbox[2] + bbox[4] / 2.) * self.input_size_height))
        #
        #     bbx_lt = (int(left * self.input_size_width), int(top * self.input_size_height))
        #     bbx_rb = (int(right * self.input_size_width), int(bottom * self.input_size_height))
        #     cv2.rectangle(img, bbx_lt, bbx_rb, (0.0, 255.0, 255.0, 255.0), 2)
        # cv2.imwrite("trb1.jpg", img)

        # augment
        if self.augment:
            dw = self.jitter_x * img_width
            dh = self.jitter_y * img_height
            new_ar = (img_width + random.uniform(-dw, dw)) / (img_height + random.uniform(-dh, dh))

            scale = 1
            if new_ar < 1:
                nh = scale * self.input_size_height
                nw = nh * new_ar
            else:
                nw = scale * self.input_size_width
                nh = nw / new_ar
            dx = random.uniform(0, self.input_size_width - nw)
            dy = random.uniform(0, self.input_size_height - nh)

            img = cv2.resize(img.copy(), (int(nw), int(nh)))

            # random_distort_image
            hue = random.uniform(-self.hue, self.hue)
            saturation = random.uniform(1, self.saturation)
            saturation = saturation if self.rand() % 2 else 1. / saturation
            exposure = random.uniform(1, self.exposure)
            exposure = exposure if self.rand() % 2 else 1. / exposure
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            H = img_hsv[:, :, 0].astype(np.float32) + hue * 180.  # hue
            S = img_hsv[:, :, 1].astype(np.float32) * saturation  # saturation
            V = img_hsv[:, :, 2].astype(np.float32) * exposure  # value
            # print("hue %f sat %f exp %f" % (hue, saturation, exposure))
            img_hsv[:, :, 0] = H.clip(0, 180)
            img_hsv[:, :, 1] = S.clip(0, 255)
            img_hsv[:, :, 2] = V.clip(0, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
            # random_distort_image
            # cv2.imwrite("trb_" + str(index) + "_" + str(self.debI) + ".jpg", img)
            T = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, T, (self.input_size_width, self.input_size_height), flags=cv2.INTER_LINEAR,
                                 borderValue=(0.5, 0.5, 0.5))

            flip = int(random.random() * 32767) % 2
            if flip:
                img = cv2.flip(img, 1)
            # cv2.imwrite("trbA_" + str(index) + "_" + str(self.batch_count) + ".jpg", img)

            dxw_dx = -dx / self.input_size_width
            dyh_dy = -dy / self.input_size_height
            nww_sx = nw / self.input_size_width
            nhh_sy = nh / self.input_size_height
            # flip, -dx/w, -dy/h, nw/w, nh/h

            nL, valid_count = len(target), 0 # valid count: the number of valid target after data augmentation.
            new_target = np.zeros((nL, 5))
            for bbox in target:
                left = bbox[0] - bbox[2] / 2.
                right = left + bbox[2]
                top = bbox[1] - bbox[3] / 2.
                bottom = top + bbox[3]

                left = nww_sx * left - dxw_dx
                right = nww_sx * right - dxw_dx
                top = nhh_sy * top - dyh_dy
                bottom = nhh_sy * bottom - dyh_dy

                if flip:
                    swap = left
                    left = 1. - right
                    right = 1. - swap

                left = self.constrain(0, 1, left)
                right = self.constrain(0, 1, right)
                top = self.constrain(0, 1, top)
                bottom = self.constrain(0, 1, bottom)
                bbox[0] = (left + right) / 2.
                bbox[1] = (top + bottom) / 2.
                bbox[2] = self.constrain(0, 1, right - left)
                bbox[3] = self.constrain(0, 1, bottom - top)
                # remove the target which is valid after data augmentation.
                if bbox[2] < 0.001 or bbox[3] < 0.001:
                    continue
                new_target[valid_count] = bbox
                valid_count += 1
                
                # bbx_lt = (int((bbox[1] - bbox[3] / 2.) * self.input_size_width),
                #           int((bbox[2] - bbox[4] / 2.) * self.input_size_height))
                # bbx_rb = (int((bbox[1] + bbox[3] / 2.) * self.input_size_width),
                #           int((bbox[2] + bbox[4] / 2.)*self.input_size_height))
                # cv2.rectangle(img, bbx_lt,
                #               bbx_rb, (0.0, 255.0, 255.0, 255.0), 2)
        else:
            img = cv2.resize(img, (int(self.input_size_width), int(self.input_size_height)))
            new_target = target
            valid_count = len(target)

        # cv2.imwrite("trbF_" + str(index) + "_" + str(self.debI) + ".jpg", img)

        labels_out = torch.zeros((130, 6))

        if valid_count:
            labels_out[:valid_count, 1:] = torch.from_numpy(new_target[:valid_count])

        img = torch.from_numpy(img).float()
        img = img.permute((2, 0, 1))
        img = torch.unsqueeze(img.div(255.0), 0)
        return img, labels_out

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """

        :param batch:
        :return:    images:     type: tensor, shape:(batch_size, channel, w, h))
                    targets:    type: list, len(targets) equal batch_size.
        """
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        self.batch_count += 1

        return torch.squeeze(torch.stack(imgs, 0), 1), torch.cat(targets, 0)



if __name__ == '__main__':
    data_path = '/home/lingc1/data/sports-training-data/player_detection/training_dataset_debug'
    data_set = YoloDataSets(data_path, image_set='train_freed_2k')
    batch_size = 2
    dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=1, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, (imgs, targets) in enumerate(dataloader):
            print(len(imgs))
            print(len(targets))
            print("===========================")

    # for i in range(1000):
    #     images, targets = next(batch_iter)
    #     print(len(images))
    #     print(len(targets))
    #     print("===========================")
