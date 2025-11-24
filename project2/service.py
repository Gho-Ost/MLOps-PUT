import bentoml
import torch
from pathlib import Path
from utils151942 import BaseLSTMForecaster

# Load model at startup
MODEL_PATH = Path("../project1/model/model.ckpt")
model = BaseLSTMForecaster.load_from_checkpoint(MODEL_PATH)
model.eval()

def preprocess_input(values, device="cuda"):
    tensor = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1).to(device)
    return tensor

@bentoml.service
class Forecasting:
    def __init__(self) -> None:
        self.model = model

    @bentoml.api
    def day_forecast(self, data) -> str:
        preprocessed_data = preprocess_input(data)
        result = self.model(preprocessed_data)
        return str(result.cpu().detach().numpy().tolist())
    
    
