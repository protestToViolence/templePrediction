import streamlit as st
import torch.nn as nn

import torch
from torchvision import models, transforms
from PIL import Image

CATEGORIES = ["AIHOLE", "BILLESHWAR_TEMPLE", "CHENNAKESHWARA_TEMPLE", "HAMPI_CHARIOT", "IBRAHIM_ROZA", "JAIN_BASADI", "KAMAL_BASTI", "KEDARESHWARA_TEMPLE", "KESHAVA_TEMPLE", "LOTUS_MAHAL"]
IMG_SIZE = 224
# Load the trained model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CATEGORIES))
model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device('cpu')))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the prediction function
def classify_image(image):
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

# Streamlit app
def main():
    st.title("Temple Image Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify image on button click
        if st.button("Classify"):
            prediction = classify_image(image)
            st.write(f"Predicted Category: {CATEGORIES[prediction]}")

if __name__ == "__main__":
    main()
