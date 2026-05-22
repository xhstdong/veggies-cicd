import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 15

def initialize_model(num_classes):
    # initalize resnet18 model
    # model weights will be loaded separately
    # generate a final fully connected layer for tailored classification
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_model(weights_path: str, device: torch.device):
    # load model weights
    model = initialize_model(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model