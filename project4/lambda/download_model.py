import torchvision.models as models
import torch
import os

def download():
    print("Downloading MobileNetV3 Small weights...")
    # Use the default weights (IMAGENET1K_V1)
    model = models.mobilenet_v3_small(weights='DEFAULT')
    torch.save(model.state_dict(), "mobilenet_v3_small.pt")
    print("Saved to mobilenet_v3_small.pt")

if __name__ == "__main__":
    download()
