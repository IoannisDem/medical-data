import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

def create_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model

def model_fn(model_dir):
    model = create_model()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
    return output.item()  # Return scalar probability

def output_fn(prediction, content_type):
    if content_type == 'text/plain':
        return str(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {content_type}")
