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
        self.init_size = cfg.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(cfg.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, cfg.img_channel, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *self.blocks(cfg.img_channel, 16, bn=False),
            *self.blocks(16, 32),
            *self.blocks(32, 64),
            *self.blocks(64, 128)
        )

        # the height and width of downsample image
        ds_size = cfg.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.adv_layer(x)
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

    # define adversarial loss
    adversarial_loss = nn.BCELoss()

    if is_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # define Optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

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

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()
            noise_img = tensor(np.random.normal(0, 1, (imgs.shape[0], cfg.latent_dim)))
            gen_imgs = generator(noise_img)

            # Loss measure generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------`
            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

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
    b1 = 0.5
    b2 = 0.999
    num_workers = 8
    latent_dim = 100
    img_size = 32
    sample_interval = 1000
    shuffle = True
    img_channel = 1


if __name__ == '__main__':
    main(Parameter())
    # test()