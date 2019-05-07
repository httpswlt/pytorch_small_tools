# coding:utf-8
def tsne(samples, labels=None, lr=100):
    """
    :param samples: ndarray, first dim is samples
    :param labels: if specified, visual img will use diff colours to mark each classes.
        ndarray with shape(n_samples)
    :param lr: lr used in TSNE
    """

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    suppressed_samples = TSNE(learning_rate=lr).fit_transform(samples.reshape(samples.shape[0], -1))
    plt.scatter(suppressed_samples[:, 0], suppressed_samples[:, 1], c=labels)
    plt.show()
    plt.waitforbuttonpress()


def test_tsne():
    import numpy as np
    from keras.datasets import cifar10

    (x_train, y_train), (_, _) = cifar10.load_data()

    idx = np.random.choice(len(x_train), 2000)
    x_train, y_train = x_train[idx], y_train[idx]

    x_train = x_train.astype('float32') / 255.
    # x_train = x_train.reshape((x_train.shape[0], -1))

    tsne(x_train, y_train)


if __name__ == '__main__':
    test_tsne()