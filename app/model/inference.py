import torch
from PIL import Image
import io
from app.model.transforms import get_transforms

LABEL_NAMES = [
    'Bean','Bitter_Gourd','Bottle_Gourd', 'Brinjal',
    'Broccoli', 'Cabbage','Capsicum', 'Carrot',
    'Cauliflower', 'Cucumber','Papaya', 'Potato',
    'Pumpkin', 'Radish','Tomato'
]

class VegetableClassifier:
    '''
    classifier class that starts a model instance
    and provides a predict method to classify input images of vegetables.
    output is a dictionary with class and confidence.
    '''
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = get_transforms()

    def predict(self, image_bytes: bytes):
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return {
            "class": LABEL_NAMES[predicted.item()],
            "confidence": float(confidence.item())
        }