import torch
import numpy as np
from torch import nn, optim
from torch import cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), latent_dim=100):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            *self.block(self.latent_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), *self.image_shape)
        return x

    @staticmethod
    def block(in_channel, out_channel, normalize=True):
        layers = [nn.Linear(in_channel, out_channel)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_channel, 0.8))
        layers.append(nn.LeakyReLU(inplace=True))
        return layers


class Discriminator(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
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
    generator = Generator(latent_dim=cfg.latent_dim)
    discriminator = Discriminator()

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
    latent_dim = 50
    img_size = 28
    sample_interval = 1000
    shuffle = True


if __name__ == '__main__':
    # main(Parameter())
    test()