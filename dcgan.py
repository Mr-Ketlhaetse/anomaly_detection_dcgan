import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class DCGAN:
    def __init__(self, dataset, nz=100, ngf=64, ndf=64, nc=3):
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.dataset = dataset

        # Initialize the networks
        self.netG = self.Generator(nz, ngf, nc)
        self.netD = self.Discriminator(nc, ndf)

        # Initialize losses dictionary
        self.losses = {'G': [], 'D': []}

        # Define the loss function and optimizers
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizerG = optim.SGD(self.netG.parameters(), lr=0.0002, momentum=0.9)
        self.optimizerD = optim.SGD(self.netD.parameters(), lr=0.0002, momentum=0.9)

    class Generator(nn.Module):
        def __init__(self, nz, ngf, nc):
            super(DCGAN.Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # Adjusted kernel size, output channels, and padding
                nn.Tanh()
            )

        def forward(self, x):
            return self.main(x)


    class Discriminator(nn.Module):
        def __init__(self, nc, ndf):
            super(DCGAN.Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            # print("input shape :", x.shape)
            if isinstance(x, list):
                # If it is, stack the tensors in the list into a single tensor
                x = torch.stack(x)
            return self.main(x)


    def train(self, num_epochs=100):
        
        dataset = self.dataset
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sigmoid Cross Entropy Loss
        criterion = self.criterion #nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
            # for i, data in dataloader:
                # Update Discriminator
                self.netD.zero_grad()
                # real_data = data[0] #.to(device)
                batch_size = data.size(0)

                # Create real labels (1s)
                label_real = torch.full((batch_size, 1), 1.0, device=device)  # Adjusted to match the dimensions

                # output_real = self.netD(real_data) #.view(-1)
                output_real = self.netD(data).mean([2, 3])
                # label_real = label_real  # .view(-1, 1)  # Reshape label_real to match the shape of output_real
                errD_real = criterion(output_real, label_real)  # Calculate the loss with the reshaped labels
                errD_real.backward()

                noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
                fake_data = self.netG(noise)

                # Create fake labels (0s)
                label_fake = torch.full((batch_size, 1), 0.0, device=device)  # Adjusted to match the dimensions

                # output_fake = self.netD(fake_data.detach()).view(-1)
                output_fake = self.netD(fake_data.detach()).mean([2, 3])
                errD_fake = criterion(output_fake, label_fake)
                errD_fake.backward()
                self.optimizerD.step()

                # Update Generator
                self.netG.zero_grad()
                label_real.fill_(1.0)  # Adjusted to use float labels
                output = self.netD(fake_data).mean([2, 3])
                errG = criterion(output, label_real)
                errG.backward()
                self.optimizerG.step()

                # Store losses
                self.losses['G'].append(errG.item())
                self.losses['D'].append(errD_real.item() + errD_fake.item())

                if i % 10 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD_real.item() + errD_fake.item(), errG.item()))

            # Save generated images for every epoch
            image_dir = "./sampled/dcgan_images"
            os.makedirs(image_dir, exist_ok=True)
            fake = self.netG(torch.randn(64, self.nz, 1, 1, device=device))
            vutils.save_image(fake.detach(), '{}/fake_samples_epoch_{}.png'.format(image_dir, epoch + 1), normalize=True)
            

        # Save the generator model
        torch.save(self.netD.state_dict(), 'dcgan_discriminator_weights.pth')
        
        # plot the loss graph for the generator and discriminator        
        plt.figure(figsize=(5,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.losses['G'],label="G")
        plt.plot(self.losses['D'],label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # Set y-axis limits here
        plt.ylim([-1, 2])  # Adjust as needed
        plt.savefig('plots/dcgan/loss_graph.png')
        plt.clf()
        pass




# if __name__ == '__main__':

#     # Example usage:
#     # my_dataset = datasets.CIFAR10  # Set your dataset class dynamically
#     transform = transforms.Compose([
#         transforms.Resize(79),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     my_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, num_workers=2)
#     dcgan = DCGAN(dataset=dataloader)
#     dcgan.train()
#     pass
    