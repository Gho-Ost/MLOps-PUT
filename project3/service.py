import bentoml
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

@bentoml.service(
    name="resnet_classifier_service",
    resources={"cpu": "1"}
)
class ResNetService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ResNet18 model on {self.device}...")
        
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        self.categories = weights.meta["categories"]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @bentoml.api
    def classify(self, img: Image.Image) -> dict:
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)
        
        probabilities = F.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        results = []
        for i in range(top5_prob.size(0)):
            class_id = int(top5_catid[i])
            results.append({
                "class_id": class_id,
                "class_name": self.categories[class_id],
                "probability": float(top5_prob[i])
            })
            
        return {"predictions": results}
