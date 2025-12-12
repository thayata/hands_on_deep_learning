import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import os

# Define hyperparameters
image_size = 32
batch_size = 64
num_epochs = 200
learning_rate = 0.0002
latent_dim = 10
num_classes = 10  # Adjust based on your dataset
embedding_dim = 5

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset (CIFAR-10 as an example)
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embedding_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(embedding_dim + int(np.prod((3, image_size, image_size))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if os.path.exists('generator_image.pth')==1:
    generator.load_state_dict(torch.load('generator_image.pth'))
    discriminator.load_state_dict(torch.load('discriminator_image.pth'))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        # Configure input
        real_imgs = imgs.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Save generated images every epoch
    if not os.path.exists('images'):
        os.makedirs('images')
    save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)

# Save the final models
torch.save(generator.state_dict(), 'generator_image.pth')
torch.save(discriminator.state_dict(), 'discriminator_image.pth')

# Mapping of class names to labels
class_to_idx = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

# Function to generate and save an image given a target name
def generate_image(target_name, generator, class_to_idx, latent_dim=100):
    generator.eval()  # Set the generator to evaluation mode
    z = torch.randn(1, latent_dim)
    label = torch.tensor([class_to_idx[target_name]], dtype=torch.long)
    gen_img = generator(z, label)
    save_image(gen_img.data, f"generated_{target_name}.png", normalize=True)
    print(f"Generated image for target '{target_name}' saved as 'generated_{target_name}.png'")

# Load the trained generator
#generator.load_state_dict(torch.load('generator.pth'))

# Example: Generate an image for a given target name (e.g., 'cat')
target_name = 'cat'  # Replace with the desired target name
generate_image(target_name, generator, class_to_idx,latent_dim=latent_dim)
