# coding:utf-8


def adjust_learning_rate(lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr_ = 1e-6 + (lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr_ = lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


def draw_image(image, coordinate):
    import cv2
    x1, y1, x2, y2 = int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3])
    cv2.putText(image, "123", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
