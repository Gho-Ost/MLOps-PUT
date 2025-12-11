import torchvision.models as models

def setup_model():
    print("Downloading/Verifying Pre-trained ResNet18 model weights...")
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("Model weights downloaded/verified successfully!")

if __name__ == "__main__":
    setup_model()
