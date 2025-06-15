
import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
from io import BytesIO

# -----------------------------
# Model Definition
# -----------------------------
import torch

# Deine eigene Modellklasse
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 56 * 56, 128),  # Achtung: 224x224 Inputgröße
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# 🔄 Lokales Modell laden
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model/model.pth", map_location=torch.device("cpu"), weights_only=False))
    model.eval()
    return model

model = load_model()


# -----------------------------
# Layout + Styling
# -----------------------------
st.set_page_config(page_title="From Old to Bold")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne&display=swap');
        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif !important;
            background-color: #ffffff;
            color: #000000;
        }
        .center-logo { display: flex; justify-content: center; margin-top: 2rem; }
        .external-button-small {
            display: flex; justify-content: center; margin-bottom: 2rem;
        }
        .external-button-small a {
            background-color: black; color: white; padding: 6px 12px;
            font-size: 0.85rem; border-radius: 6px; text-decoration: none;
        }
        .description-text {
            text-align: center; font-size: 1.1rem; margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    cols = st.columns(3)
    with cols[1]:
        st.image("logo.png", width=180)

st.markdown('<div class="description-text">Upload a photo of your old piece of jewelry. Our AI estimates the weight and suggests matching new designs!</div>', unsafe_allow_html=True)

st.markdown("""
<div class="external-button-small">
    <a href="https://eager-transform-667249.framer.app/" target="_blank">WHAT IS FROM OLD TO BOLD</a>
</div>
""", unsafe_allow_html=True)

material = st.selectbox("Select material", ["Silver", "Gold", "Other"])
if material == "Other":
    custom_material = st.text_input("Please specify the material")
    if custom_material:
        material = custom_material

uploaded_file = st.file_uploader("Upload an image of your old jewelry", type=["jpg", "jpeg", "png"])

# -----------------------------
# Prediction
# -----------------------------
def predict_weight(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    return prediction.item()


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    weight = predict_weight(image)
    st.write(f"**Estimated weight:** {weight:.2f} grams")

    df = pd.read_csv("designs.csv")
    tolerance = 0.7
    matched = df[
        (abs(df["weight"] - weight) <= tolerance) &
        (df["material"].str.lower() == material.lower())
    ]

    st.subheader("Matching designs:")
    if not matched.empty:
        for _, row in matched.iterrows():
            st.image(row["filename"], caption=f"[{row['name']}]({row['link']}) – {row['weight']} g", use_container_width=True)
    else:
        st.write("No matching designs found.")
