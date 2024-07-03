import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Define the generator network using a pre-trained VGG model
class Generator(nn.Module):
    def __init__(self, nz=512 * 2 * 2, nc=3):
        super(Generator, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.fc = nn.Sequential(
            nn.Linear(nz, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = self.vgg(input)
        input = input.view(-1, nz)
        input = self.fc(input)
        input = input.view(-1, 512, 8, 8)
        return self.upsample(input)

# Define the discriminator network using transfer learning (pre-trained ResNet)
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.resnet(input)

# Initialize the weights of the discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Hyperparameters
batch_size = 64
image_size = 64
nz = 512*2*2 # size of the latent z vector
nc = 3  # number of channels in the training images (for RGB)
ngf = 64  # size of feature maps in the generator
num_epochs = 1
lr = 0.0002
beta1 = 0.5

# Data loading
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the generator and discriminator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, nc).to(device)
netD = Discriminator(nc).to(device)

# Initialize the discriminator weights
netD.apply(weights_init)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary cross-entropy loss
criterion = nn.BCELoss()

# Fixed noise for generating images
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator network
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1., dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, 3, 3, 3, device=device)
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator network
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Print statistics
        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

    # Save generated images
    vutils.save_image(real_cpu, f'results/real_samples_epoch_{epoch}.png', normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), f'results/fake_samples_epoch_{epoch}.png', normalize=True)

# Save the models
#torch.save(netG.state_dict(), 'generator.pth')
#torch.save(netD.state_dict(), 'discriminator.pth')

def generate_images(epoch, generator, latent_dim, num_images=10, image_size=(nc, image_size, image_size)):
    z = torch.randn(num_images, latent_dim).to(device)
    fake_images = generator(z)
    fake_images = fake_images.view(fake_images.size(0), *image_size)
    fake_images = (fake_images + 1) / 2.0  # Rescale to [0, 1]
    fake_images = fake_images.cpu().detach().numpy()
        
    fig, axes = plt.subplots(5, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(fake_images[i], (1, 2, 0)))
        #ax.imshow(np.transpose(npimg, (1, 2, 0)) if dataset_name == 'CIFAR10' else npimg[0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'{sample_dir}/generated_images_epoch_{epoch}.png')
    #plt.close()

# Generate new images
generate_images(epoch+1, netG.state_dict(), nz)