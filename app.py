import gradio as gr
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = not feature_extracting

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def predict_vegetable(image):
    # image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_ft(image)
        _, predicted = torch.max(outputs, 1)
    return label_names[predicted.item()]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 15
    feature_extract = True

    # Load the trained model
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=False) #use_pretrained=False to use our own weights
    model_ft.load_state_dict(torch.load('veggie_model_weights.pth', map_location=torch.device('cpu'))) #add map_location here
    model_ft.eval()
    model_ft.to(device)

    # Define the label names (make sure this matches your training data)
    label_names = ['Bean','Bitter_Gourd','Bottle_Gourd', 'Brinjal',
                   'Broccoli', 'Cabbage','Capsicum', 'Carrot',
                   'Cauliflower', 'Cucumber','Papaya', 'Potato',
                     'Pumpkin', 'Radish','Tomato']

    # Image preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define prediction action
    def predict_vegetable(image):
        if image is None:
            return "Please upload an image."
        # image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_ft(image)
            _, predicted = torch.max(outputs, 1)
        if 0 <= predicted.item() < len(label_names):
            return label_names[predicted.item()]
        else:
            return "Unknown"

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict_vegetable,
        inputs=gr.Image(type="pil"), # Changed from gr.inputs.Image to gr.Image
        outputs=gr.Label(num_top_classes=1), # Changed from gr.outputs.Label to gr.Label
        title="Vegetable Image Classifier",
        description="Upload an image of a vegetable to classify it.",
    )

    iface.launch(debug=True)