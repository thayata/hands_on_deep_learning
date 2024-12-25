import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Load pre-trained Stable Diffusion model components
model_id = "CompVis/stable-diffusion-v1-4"
device = "mps" if torch.cuda.is_available() else "cpu"

# Load the UNet model and other components
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Set model to training mode
unet.train()

# Training settings
batch_size = 4
epochs = 5
learning_rate = 1e-5

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Example dataset, replace with your custom dataset
train_dataset = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and Loss Function
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Define a function to encode text prompts
def encode_prompts(prompts):
    inputs = tokenizer(prompts, padding='max_length', max_length=77, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    return text_encoder(input_ids)[0]

# Training Loop
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # Generate random text prompts (for example purposes)
        prompts = ["A photo of a cat"] * data.size(0)
        text_embeddings = encode_prompts(prompts)

        # Forward pass through the UNet model
        noise = torch.randn_like(data).to(device)
        timesteps = torch.randint(0, 1000, (data.size(0),), device=device).long()
        noisy_data = data * (1 - timesteps.unsqueeze(1).unsqueeze(2).unsqueeze(3) / 1000) + noise * (timesteps.unsqueeze(1).unsqueeze(2).unsqueeze(3) / 1000)
        
        optimizer.zero_grad()
        outputs = unet(noisy_data, timesteps, text_embeddings).sample

        # Compute loss and backpropagation
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item()}')

print("Fine-tuning complete.")

# Save the fine-tuned model
unet.save_pretrained("fine_tuned_unet")

# Image generation using the fine-tuned model
def generate_image(model, prompt, steps=50, image_size=(3, 512, 512)):
    model.eval()
    with torch.no_grad():
        # Encode the text prompt
        text_embeddings = encode_prompts([prompt])
        
        # Generate random noise
        x = torch.randn((1, *image_size)).to(device)
        for step in range(steps, 0, -1):
            t = torch.tensor([step / steps], dtype=torch.float32).view(1).to(device)
            x = model(x, t, text_embeddings).sample
    return x

# Generate and save an image
generated_image = generate_image(unet, "A beautiful landscape with mountains and a river")
save_image(generated_image, "generated_image.png")
print("Generated image saved as generated_image.png")
