import os
import torch
import numpy as np
from torch import nn, optim
from torch import cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)


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
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        # Downsampling
        self.down = nn.Sequential(nn.Conv2d(cfg.img_channel, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = cfg.img_size // 2
        down_dim = 64 * (cfg.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, cfg.img_channel, 3, 1, 1))

    def forward(self, x):
        x = self.down(x)
        x = self.fc(x.view(x.size(0), -1))
        x = self.up(x.view(x.size(0), 64, self.down_size, self.down_size))
        return x


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

    # initialize generator and discriminator weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # define Optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    # began hyper parameters
    gamma = 0.75
    lambda_k = 0.001
    k = 0.0
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
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()
            gen_imgs = generator(noise_imgs)
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))
            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            #  Update weights
            # ---------------------
            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()

            # ---------------------
            #  Log Progress
            # ---------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, cfg.epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
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
    clip_value = 5


if __name__ == '__main__':
    main(Parameter())
    # test()
