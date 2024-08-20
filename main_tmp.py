import torch
from flwr_datasets import FederatedDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision import models

from fedavg_mobilne.fedavg_mobilnet.utils import initialize_weights


def apply_transforms(batch):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply transformations
    batch["image"] = [tf(img) for img in batch["image"]]
    return batch



dataset = FederatedDataset(dataset="hf-vision/chest-xray-pneumonia", partitioners={"train": 1})

client_dataset = dataset.load_partition(0)


client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)



train_dataset = client_dataset_splits["train"]
val_dataset = client_dataset_splits["test"]


train_dataset = train_dataset.with_transform(apply_transforms)
val_dataset = val_dataset.with_transform(apply_transforms)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

####

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

num_classes = 2
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.apply(initialize_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#####

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the validation images: {100 * correct / total:.2f}%')

