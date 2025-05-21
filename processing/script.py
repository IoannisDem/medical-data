import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_dir = Path(os.getenv("input_path"))
    output_dir = Path(os.getenv("output_path"))

    for img_path in input_dir.glob("*.jpg"):
        with Image.open(img_path).convert("RGB") as img:
            transformed_tensor = transform(img)
            # Convert tensor back to PIL (unnormalize first)
            unnormalize = transforms.Normalize(mean=[-1]*3, std=[1/0.5]*3)
            pil_img = transforms.ToPILImage()(unnormalize(transformed_tensor))
            pil_img.save(output_dir / img_path.name)

if __name__ == "__main__":
    main()
