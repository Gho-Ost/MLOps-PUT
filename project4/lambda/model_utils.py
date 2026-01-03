import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Global variable to cache the model in memory across warm invocations
MODEL = None

def load_model(model_path="mobilenet_v3_small.pt"):
    global MODEL
    if MODEL is None:
        print(f"Loading model from {model_path}...")
        # Initialize architecture
        model = models.mobilenet_v3_small(weights=None)
        
        # Load state dict
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Model weights loaded.")
        else:
             print(f"WARNING: Model weights not found at {model_path}. Using random initialization (will output garbage).")
        
        model.eval()
        MODEL = model
    return MODEL

def predict(image_path, model_path="mobilenet_v3_small.pt"):
    model = load_model(model_path)
    
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    
    # Get top 1
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    
    return int(top1_catid[0])
