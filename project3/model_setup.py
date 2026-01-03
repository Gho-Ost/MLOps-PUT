import torchvision.models as models

def setup_model():
    print("Downloading/Verifying pre-trained MobileNetV3 Small model weights...")
    # Prefer the new weights enum when available, fall back to older names or pretrained flag
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    except AttributeError:
        try:
            weights = models.MobileNetV3_Small_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None

    if weights is not None:
        models.mobilenet_v3_small(weights=weights)
    else:
        # Older torchvision versions use the `pretrained=True` argument
        models.mobilenet_v3_small(pretrained=True)

    print("Model weights downloaded/verified successfully!")

if __name__ == "__main__":
    setup_model()
