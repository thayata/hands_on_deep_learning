import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
image_size = 48
batch_size = 32
learning_rate = 0.0002
layer_dim = 256
num_epochs = 20
latent_dim = 50

# Device configuration
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# MNIST dataset changed to CIFAR-10
mnist = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(layer_dim, 2*layer_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2*layer_dim, 4*layer_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4*layer_dim, 3*image_size * image_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, image_size, image_size)
        return x

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod((3, image_size, image_size))), 4*layer_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(4*layer_dim, 2*layer_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2*layer_dim, layer_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(layer_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

# Initialize models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training the GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        #print(images.shape)

        # Train Discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(images.to(device))
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')

# Save the models
#torch.save(generator.state_dict(), 'generator.pth')
#torch.save(discriminator.state_dict(), 'discriminator.pth')


def generate_images(epoch, generator, latent_dim, num_images=10, image_size=(3, image_size, image_size)):
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
generate_images(epoch+1, generator, latent_dim)