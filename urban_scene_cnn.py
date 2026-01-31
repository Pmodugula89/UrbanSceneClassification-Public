import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

# -----------------------------
# Step 1: Dataset Preparation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # smaller image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Download CIFAR-10 and use only first 1000 samples for speed
full_train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Reduce dataset size
subset_train_set = Subset(full_train_set, range(1000))
subset_test_set = Subset(test_set, range(200))

# Split train into train/val
train_size = int(0.8 * len(subset_train_set))
val_size = len(subset_train_set) - train_size
train_set, val_set = random_split(subset_train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(subset_test_set, batch_size=32, shuffle=False)

# Display sample image
sample_image, sample_label = full_train_set[0]
plt.imshow(sample_image.permute(1, 2, 0))
plt.title(f"Sample Image - Class {full_train_set.classes[sample_label]}")
plt.axis("off")
plt.show()

# -----------------------------
# Step 2: CNN Model Definition
# -----------------------------
class UrbanSceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(UrbanSceneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)  # adjusted for 64x64 input

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

num_classes = len(full_train_set.classes)
model = UrbanSceneCNN(num_classes)
print(model)

# -----------------------------
# Step 3: Training Function
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if total > 0:
            val_accuracy = correct / total
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}, No validation samples available.")

train_model(model, train_loader, val_loader, optimizer, criterion)

# -----------------------------
# Step 4: Evaluation
# -----------------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

plt.bar(["Test Accuracy"], [test_accuracy])
plt.ylabel("Accuracy")
plt.title("CNN Model Performance")
plt.show()