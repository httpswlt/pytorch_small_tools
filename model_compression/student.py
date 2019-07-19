# coding:utf-8
import itertools
import os
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import utils
import networks
import matplotlib.pyplot as plt


def prepare_data():
    mnist_image_shape = (28, 28)

    random_pad_size = 2
    # Training images augmented by randomly shifting images by at max. 2 pixels in any of 4 directions
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(mnist_image_shape, random_pad_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    train_val_dataset = torchvision.datasets.MNIST(root='~/datasets', train=True,
                                                   download=False, transform=transform_train)

    test_dataset = torchvision.datasets.MNIST(root='~/datasets', train=False,
                                              download=False, transform=transform_test)
    num_train = int(1.0 * len(train_val_dataset) * 95 / 100)
    num_val = len(train_val_dataset) - num_train
    train_dataset, val_dataset = data.random_split(train_val_dataset, [num_train, num_val])

    # batch_size = 128
    train_val_loader = data.DataLoader(train_val_dataset, batch_size=128, shuffle=True, num_workers=2)

    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return train_val_loader, train_loader, val_loader, test_loader


def train():
    train_val_loader, train_loader, val_loader, test_loader = prepare_data()
    checkpoints_path_teacher = 'checkpoints_teacher/'
    checkpoints_path_student = 'checkpoints_student/'
    if not os.path.exists(checkpoints_path_student):
        os.mkdir(checkpoints_path_student)

    fast_device = torch.device('cuda:' + str(0))
    num_epochs = 60
    print_every = 100  # Interval size for which to print statistics of training
    learning_rates = [1e-2]
    learning_rate_decays = [0.95]  # learning rate decays at every epoch
    # weight_decays = [0.0] + list(np.logspace(-5, -1, 5))
    weight_decays = [1e-5]  # regularization weight
    momentums = [0.9]
    # dropout_probabilities = [(0.2, 0.5), (0.0, 0.0)]
    dropout_probabilities = [(0.0, 0.0)]
    hparams_list = []
    for hparam_tuple in itertools.product(dropout_probabilities, weight_decays, learning_rate_decays,
                                          momentums, learning_rates):
        hparam = dict()
        hparam['dropout_input'] = hparam_tuple[0][0]
        hparam['dropout_hidden'] = hparam_tuple[0][1]
        hparam['weight_decay'] = hparam_tuple[1]
        hparam['lr_decay'] = hparam_tuple[2]
        hparam['momentum'] = hparam_tuple[3]
        hparam['lr'] = hparam_tuple[4]
        hparams_list.append(hparam)
    load_path = checkpoints_path_teacher + utils.hparamToString(hparams_list[0]) + '_final.tar'
    teacher_net = networks.TeacherNetwork()
    teacher_net.load_state_dict(torch.load(load_path, map_location=fast_device)['model_state_dict'])
    teacher_net = teacher_net.to(fast_device)
    # Calculate teacher test accuracy
    _, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, fast_device)
    print('teacher test accuracy: ', test_accuracy)

    # # # # # # # # # # # # # #
    #   training student      #
    # # # # # # # # # # # # # #
    temperatures = [10]  # temperature for distillation loss
    # trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
    # loss = alpha * st + (1 - alpha) * tt
    alphas = [0.5]
    learning_rates = [1e-2]
    learning_rate_decays = [0.95]
    weight_decays = [1e-5]
    momentums = [0.9]
    # No dropout used
    dropout_probabilities = [(0.0, 0.0)]
    hparams_list = []
    for hparam_tuple in itertools.product(alphas, temperatures, dropout_probabilities, weight_decays,
                                          learning_rate_decays,
                                          momentums, learning_rates):
        hparam = dict()
        hparam['alpha'] = hparam_tuple[0]
        hparam['T'] = hparam_tuple[1]
        hparam['dropout_input'] = hparam_tuple[2][0]
        hparam['dropout_hidden'] = hparam_tuple[2][1]
        hparam['weight_decay'] = hparam_tuple[3]
        hparam['lr_decay'] = hparam_tuple[4]
        hparam['momentum'] = hparam_tuple[5]
        hparam['lr'] = hparam_tuple[6]
        hparams_list.append(hparam)

    results_no_distill = {}
    for hparam in hparams_list:
        print('Training with hparams' + utils.hparamToString(hparam))
        reproducibilityseed()
        student_net = networks.StudentNetwork()
        student_net = student_net.to(fast_device)
        hparam_tuple = utils.hparamDictToTuple(hparam)
        results_no_distill[hparam_tuple] = utils.trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs,
                                                                      train_val_loader, None,
                                                                      print_every=print_every,
                                                                      fast_device=fast_device)
        save_path = checkpoints_path_student + utils.hparamToString(hparam) + '_final.tar'
        torch.save({'results': results_no_distill[hparam_tuple],
                    'model_state_dict': student_net.state_dict(),
                    'epoch': num_epochs}, save_path)


def reproducibilityseed(use_gpu=True):
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    train()


if __name__ == '__main__':
    main()
