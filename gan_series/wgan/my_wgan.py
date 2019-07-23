import os
import torch
import numpy as np
from torch import nn, optim
from torch import cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.img_shape = cfg.img_shape
        self.model = nn.Sequential(
            *self.block(cfg.latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], *self.img_shape)
        return x

    @staticmethod
    def block(in_channels, out_channels, normalize=True):
        layers = [nn.Linear(in_channels, out_channels)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_channels, 0.8))
        layers.append(nn.LeakyReLU(inplace=True))
        return layers


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.img_shape = cfg.img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x

    @staticmethod
    def blocks(in_channel, out_channel, bn=True):
        layers = [nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                  nn.LeakyReLU(inplace=True),
                  nn.Dropout2d(0.25)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channel, 0.8))
        return layers


def main(cfg):
    # judge whether enable cuda
    is_cuda = True if cuda.is_available() else False

    # prepare data to training
    dataloader = DataLoader(
        dataset=datasets.MNIST(
            "~/jobs/data/mnist",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(cfg.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    # define Discriminator and Generator
    generator = Generator(cfg)
    discriminator = Discriminator(cfg)

    if is_cuda:
        generator.cuda()
        discriminator.cuda()

    # define Optimizer
    optimizer_g = optim.RMSprop(generator.parameters(), lr=cfg.lr)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=cfg.lr)

    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    for epoch in range(cfg.epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # adversarial ground truths
            valid = tensor(imgs.size(0), 1).fill_(1.0)
            fake = tensor(imgs.size(0), 1).fill_(0.0)

            # real image
            real_imgs = imgs.type(tensor)
            # generator to generate images from these noise images.
            noise_imgs = tensor(np.random.normal(0, 1, (imgs.shape[0], cfg.latent_dim)))

            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_d.zero_grad()
            fake_imgs = generator(noise_imgs).detach()
            d_loss = - torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
            d_loss.backward()
            optimizer_d.step()

            # clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-cfg.clip_value, cfg.clip_value)

            if i % cfg.n_critic == 0:
                # ---------------------
                #  Train Generator
                # ---------------------
                optimizer_g.zero_grad()
                gen_imgs = generator(noise_imgs)
                g_loss = - torch.mean(discriminator(gen_imgs))
                g_loss.backward()
                optimizer_g.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, cfg.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % cfg.sample_interval == 0:
                if not os.path.exists('./images') and not os.path.exists('./model'):
                    os.makedirs('./images')
                    os.makedirs('./model')
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                torch.save(generator, "./model/generator.pth")
                torch.save(discriminator, "./model/discriminator.pth")


def test():
    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator = torch.load('./model/generator.pth')
    for j in range(3):
        noise_img = tensor(np.random.normal(0, 1, (28, 50)))
        gen_imgs = generator(noise_img)
        save_image(gen_imgs.data[:25], "test_{}.png".format(j), nrow=5, normalize=True)


class Parameter:
    epochs = 200
    batch_size = 64
    lr = 0.0002
    num_workers = 8
    latent_dim = 100
    img_size = 32
    sample_interval = 1000
    shuffle = True
    img_channel = 1
    clip_value = 5


if __name__ == '__main__':
    main(Parameter())
    # test()
