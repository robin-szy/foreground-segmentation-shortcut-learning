import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# Config
# -----------------------
num_samples = 100
num_classes = 10
batch_size = 32
epochs = 3
lr = 0.001

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Fake dataset
# -----------------------
X = torch.randn(num_samples, 3, 128, 128)
y = torch.randint(0, num_classes, (num_samples,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------
# Simple CNN
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
        return self.fc(self.conv(x))

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training
# -----------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if epoch == 0 and batch_idx == 0:
            print("Batch shape:", images.shape)
            print("Labels shape:", labels.shape)

        images = images.to(device)
        labels = labels.to(device)

        if epoch == 0 and batch_idx == 0:
            print("Images device:", images.device)
            print("Labels device:", labels.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "test_model.pth")
print("Training finished")