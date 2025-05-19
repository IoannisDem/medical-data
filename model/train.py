import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, len(train_dataset.classes)

def create_model(num_classes=1):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid() 
    )
    return model

def save_model(model, model_dir):
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

def model_fn(model_dir):
    model = create_model()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    train_loader, _ = get_data_loaders(args.data_dir, args.batch_size)
    model = create_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Reshape for BCE

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.4f}")

    save_model(model, args.model_dir)

if __name__ == '__main__':
    main()
