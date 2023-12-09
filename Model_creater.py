import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torchvision import datasets, transforms
import cv2
import mediapipe as mp
from PIL import Image
awake_count = 0
sleep_count = 0
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to match LeNet input size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# Create datasets from folders
train_dataset = datasets.ImageFolder(root = './Train_cases',transform=transform)
test_dataset = datasets.ImageFolder(root = './Gray_Sleep/Gray_sleep_test',transform=transform)

# Create data loaders for batch processing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#Training loop
num_epochs = 10
#print("kafil")
for epoch in range(num_epochs):
    running_loss = 0.0
    #print("kafil 1")
    for i, data in enumerate(train_loader, 0):
        #print(i)
        #print(train_loader(i))
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == len(train_loader)-1:  # Print every 100 mini-batches
            #print("kafil 3")
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Step [{i + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Finished Training")
torch.save(model.state_dict(), "./Final_model.pth")

