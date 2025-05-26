import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size, torch_transform):
    train_dataset = datasets.ImageFolder(root=data_dir, transform=torch_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, len(train_dataset.classes)

def create_model(num_classes=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)  # Assuming binary classification (Healthy vs Tumor)
    )
    model = model.to(device)
    return model

def save_model(model, model_dir):
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--data-test-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))


    args = parser.parse_args()
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])  

    train_loader, _ = get_data_loaders(args.data_train_dir, args.batch_size, transform_train)
    test_loader, _ = get_data_loaders(args.data_test_dir, args.batch_size, transform_test)
    model = create_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    model.train()
    early_stop_patience = 7
    best_acc = 0
    patience_counter = 0
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct, total = 0, 0
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        val_acc = 100 * correct / total
        scheduler.step(val_acc)
            
        # Early Stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            break
    
    save_model(model, args.model_dir)

if __name__ == '__main__':
    main()
