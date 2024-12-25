import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Device configuration
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


# Define a simple U-Net-like architecture for the diffusion model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the forward and reverse diffusion process
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.unet = UNet()

    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        return torch.sqrt(1 - t) * x + torch.sqrt(t) * noise

    def reverse_diffusion(self, x, t):
        return self.unet(x)

# Instantiate the model
model = DiffusionModel().to(device)

# Training settings
batch_size = 64
epochs = 100
learning_rate = 1e-4

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        t = torch.rand(data.size(0), 1, 1, 1)  # Random timesteps
        data = data.to(device)
        t = t.to(device)

        # Forward diffusion
        noisy_data = model.forward_diffusion(data, t)

        # Reverse diffusion
        reconstructed_data = model.reverse_diffusion(noisy_data, t)

        # Compute loss
        loss = criterion(reconstructed_data, data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item()}')

print("Training complete.")

# Image generation from the trained model
def generate_image(model, steps=1000, image_size=(3, 32, 32)):
    model.eval()
    with torch.no_grad():
        x = torch.randn((1, *image_size)).to(device)
        for step in range(steps, 0, -1):
            t = torch.tensor([step / steps], dtype=torch.float32).view(1, 1, 1, 1).to(device)
            x = model.reverse_diffusion(x, t)
    return x

# Create directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Generate and save images
for i in range(10):  # Generate 10 images
    generated_image = generate_image(model)
    save_image(generated_image, f"generated_images/image_{i + 1}.png")
    print(f"Generated image_{i + 1}.png saved.")

print("Image generation complete.")
