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

"""
Debugging checks:
- Add inside loop once: print(images.shape) -> Should return [32, 3, 224, 224]
- During training, run "nvidia-smi" -> memory usage should increase
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# -------------------
# Config
# -------------------

train_dir = "/home/manzana/Downloads/dataset/inaturalist_12K/train"
val_dir   = "/home/manzana/Downloads/dataset/inaturalist_12K/val"
batch_size = 32
epochs = 2
lr = 0.001
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------
# Dataset
# -------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

num_classes = len(dataset.classes)
print("Classes:", num_classes)

dataset = Subset(dataset, range(100))
val_dataset = Subset(val_dataset, range(20))

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True     # Speeds up GPU data transfer on clusters
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)


print("Images:", len(dataset))

# -------------------
# Model
# -------------------

model = models.resnet18(weights="IMAGENET1K_V1")

# replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -------------------
# Training setup
# -------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------
# Training loop
# -------------------

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation accuracy: {accuracy:.2f}%")

# -------------------
# Save model
# -------------------

torch.save(model.state_dict(), "resnet18_test.pth")

print("Training complete")