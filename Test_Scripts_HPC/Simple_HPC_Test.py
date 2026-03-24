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


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

max_images = 10000   # small smoke test

# -----------------------
# Configuration
# -----------------------
data_root = os.environ.get("DATA_ROOT", "/tmp/inaturalist_12K")
train_dir = os.path.join(data_root, "train")

print("DATA_ROOT:", os.environ.get("DATA_ROOT"))
print("Exists:", os.path.exists(train_dir))

print("=== PyTorch GPU Test ===")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    print(f"Memory allocated: {torch.cuda.memory_allocated(0)}")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)}")

    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
else:
    print("No GPU detected.")

batch_size = 32
epochs = 3
lr = 0.001
num_workers = 4
output_model = "test_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Training data path:", train_dir)

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
print("Original dataset size:", len(dataset))

if len(dataset) > max_images:
    dataset = Subset(dataset, range(max_images))
    print("Subset dataset size:", len(dataset))

pin_memory = torch.cuda.is_available()

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0)
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
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if epoch == 0 and batch_idx == 0:
            print("Batch shape:", images.shape)
            print("Labels shape:", labels.shape)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if epoch == 0 and batch_idx == 0:
            print("Images device:", images.device)
            print("Images dtype:", images.dtype)
            print("Labels device:", labels.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), output_model)
print(f"Training finished, model saved to {output_model}")