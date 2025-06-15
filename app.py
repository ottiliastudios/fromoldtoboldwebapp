import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from io import BytesIO
import requests

# App-Konfiguration
st.set_page_config(page_title="From Old to Bold")

# --- Custom Style ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne&display=swap');

        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif !important;
            background-color: #ffffff;
            color: #000000;
        }

        .center-logo {
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }

        .external-button-small {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }

        .external-button-small a {
            background-color: black;
            color: white;
            padding: 6px 12px;
            font-size: 0.85rem;
            border-radius: 6px;
            text-decoration: none;
            font-family: 'Syne', sans-serif;
        }

        .description-text {
            text-align: center;
            font-size: 1.1rem;
            font-family: 'Syne', sans-serif;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Logo ---
with st.container():
    cols = st.columns(3)
    with cols[1]:
        st.image("logo.png", width=180)

# --- Beschreibung ---
st.markdown('<div class="description-text">Upload a photo of your old piece of jewelry. Our AI estimates the weight and suggests matching new designs!</div>', unsafe_allow_html=True)

st.markdown("""
<div class="external-button-small">
    <a href="https://eager-transform-667249.framer.app/" target="_blank">WHAT IS FROM OLD TO BOLD</a>
</div>
""", unsafe_allow_html=True)

# Auswahl: Material
material = st.selectbox("Select material", ["Silver", "Gold", "Other"])
if material == "Other":
    custom_material = st.text_input("Please specify the material")

# Bild-Upload
uploaded_file = st.file_uploader("Upload an image of your old jewelry", type=["jpg", "jpeg", "png"])

# Modellklasse
def create_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 64 * 64, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    return model

@st.cache_resource
def load_model():
    model = create_model()
    model_path = "model.pth"  # Stelle sicher, dass diese Datei vorhanden ist
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
    model.eval()
    return model

# Gewicht vorhersagen
def predict_weight(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output.item()

# Modell laden
model = load_model()

# Vorhersage + Vorschläge
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    weight = predict_weight(image, model)
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
            st.image(row["filename"], caption=f"[{row['name']}]({row['url']}) – {row['weight']} g", use_container_width=True)
    else:
        st.write("No matching designs found.")
