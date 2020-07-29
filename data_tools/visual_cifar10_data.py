# coding:utf-8
import cPickle
import cv2
import numpy as np
import os


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data_dict = cPickle.load(f)
    return data_dict


def save_file(data_dict, save_path):
    datas = data_dict.get('data')
    file_names = data_dict.get('filenames')
    labels = data_dict.get('labels')
    for data, name, label in zip(datas, file_names, labels):
        data = np.reshape(data, [3, 32, 32])
        img = cv2.merge([data[2], data[1], data[0]])
        path = os.path.join(save_path, str(label))
        if not os.path.exists(path):
            os.makedirs(path)
        img_path = os.path.join(path, name)
        cv2.imwrite(img_path, img)


def main():
    root_path = '~/datasets/cifar-10-batches-py'
    save_path = '~/datasets/cifar10'

    for f in os.listdir(root_path):
        if '_batch' not in f:
            continue

        if 'test' in f:
            path = os.path.join(save_path, 'test')
        else:
            path = os.path.join(save_path, 'train')

        batch_file_path = os.path.join(root_path, f)
        save_file(unpickle(batch_file_path), path)


if __name__ == '__main__':
    main()
