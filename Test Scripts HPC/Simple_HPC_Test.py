"""
This is just a simple script to train a model for the HPC spring school that I've attended.
It is really just for testing and has nothing to do with the actual project. I just wanted to make sure that I can train a model on the HPC cluster and save it without any issues.
I've used ChatGPT (version GPT-5.3) to create the code with the following prompt:
"I have an inaturalist data set where I need to create a deep learning algorithm on. Tomorrow I have an HPC introduction training to learn how to use our HPC server properly.
For this, I already uploaded the dataset to my home directory at the HPC account. For the training tomorrow, I really only have one goal: I want to learn how to train my model
with the help of the HPC server. Nothing more. For this, I will need a small test model that does not even have to be a good model. It just needs to be trained so I can test it
on the HPC. Later, I will create a proper model, and train it by the same procedure. Can you propose a Python code to train a model on a large dataset of images? Preferably not
to computationally heavy. Just training some weights. Very basic model."
It proposed the code below.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

max_images = 100   # choose something small

# -----------------------
# Configuration
# -----------------------
train_dir = "/home/manzana/Downloads/dataset/inaturalist_12K/train"
val_dir   = "/home/manzana/Downloads/dataset/inaturalist_12K/val"

batch_size = 32
epochs = 3
lr = 0.001
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Dataset
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=train_dir, transform=transform)

num_classes = len(dataset.classes)
print("Classes:", num_classes)

# For testing on PC
if len(dataset) > max_images:
    dataset = Subset(dataset, range(max_images))

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# -----------------------
# Simple CNN Model
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = SimpleCNN(num_classes).to(device)

# -----------------------
# Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "test_model.pth")

print("Training finished")