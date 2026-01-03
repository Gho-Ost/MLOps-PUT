import bentoml
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

@bentoml.service(
    name="mobilenet_v3_small_classifier_service",
    resources={"cpu": "1"}
)
class MobileNetV3Service:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MobileNetV3 Small model on {self.device}...")

        # Prefer the new weights enum names; provide fallbacks for older torchvision
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        except AttributeError:
            try:
                weights = models.MobileNetV3_Small_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = None

        if weights is not None:
            # Use the recommended transforms when available
            try:
                self.preprocess = weights.transforms()
            except Exception:
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

            self.model = models.mobilenet_v3_small(weights=weights)
            self.categories = weights.meta.get("categories", [str(i) for i in range(1000)])
        else:
            # Fallback for very old torchvision versions
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.model.eval()
            self.categories = [str(i) for i in range(1000)]

            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.model.to(self.device)
        self.model.eval()

    @bentoml.api
    def classify(self, img: Image.Image) -> dict:
        input_tensor = self.preprocess(img)
        # If preprocess returns a batch or tensor, ensure correct shape
        if isinstance(input_tensor, torch.Tensor):
            input_batch = input_tensor.unsqueeze(0).to(self.device)
        else:
            input_batch = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)

        if isinstance(output, torch.Tensor):
            logits = output[0]
        else:
            logits = output

        probabilities = F.softmax(logits, dim=0)
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
