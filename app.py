import streamlit as st
st.set_page_config(page_title="From Old to Bold")
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ---------- SEITE KONFIGURATION ----------


# ---------- STIL ----------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne&display=swap');

        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif !important;
            background-color: #ffffff;
            color: #000000;
        }

        .original-price {
            text-decoration: line-through;
            color: gray;
            font-size: 0.9rem;
            font-family: 'Syne', sans-serif !important;
        }

        .discounted-price {
            color: green;
            font-weight: bold;
            font-size: 1rem;
            font-family: 'Syne', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- LOGO ----------
cols = st.columns([1, 1, 1])
with cols[1]:
    st.image("logo.png", width=180)

# ---------- BESCHREIBUNG ----------
st.markdown(
    """
    <div style='text-align: center; font-size: 1.1rem; margin-bottom: 2rem; font-family: "Syne", sans-serif !important;'>
        Upload a photo of your old piece of jewelry next to a ruler. Our AI estimates the weight and suggests matching new designs!
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="display: flex; justify-content: center; margin-bottom: 2rem;">
    <a href="https://eager-transform-667249.framer.app/" target="_blank" style="background-color: black; color: white; padding: 6px 12px; font-size: 0.85rem; border-radius: 6px; text-decoration: none;">WHAT IS FROM OLD TO BOLD</a>
</div>
""", unsafe_allow_html=True)

# ---------- MATERIAL AUSWAHL ----------
material = st.selectbox("Select material", ["Silver", "Gold", "Other"])
if material == "Other":
    custom_material = st.text_input("Please specify the material")
    material = custom_material

# ---------- DATEIUPLOAD ----------
uploaded_file = st.file_uploader("Upload an image of your old jewelry", type=["jpg", "jpeg", "png"])

# ---------- MODELL ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze()

@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu"), weights_only=False))
    model.eval()
    return model

model = load_model()

# ---------- PREDICT & DESIGN ANZEIGEN ----------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded image", width=200)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gewicht vorhersagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        weight = model(img_tensor).item()
    st.markdown(f"<p style='text-align: center; font-weight: bold; font-family: 'Syne', sans-serif !important;'>Estimated weight: {weight:.2f} g</p>", unsafe_allow_html=True)

    # Rabattlogik
    if material.lower() == "silver":
        discount_rate = 0.10
    elif material.lower() == "gold":
        discount_rate = 0.20
    else:
        discount_rate = 0.0

    # Designs filtern
    df = pd.read_csv("designs.csv", sep=";")
    tolerance = 1.0
    matched = df[
        (abs(df["weight"] - weight) <= tolerance) &
        (df["material"].str.lower() == material.lower())
    ]

    # Designs anzeigen
    if not matched.empty:
        st.markdown("<h4 style='margin-left: 16px;'>Matching Designs:</h4>", unsafe_allow_html=True)

        html_gallery = "<div style='display: flex; flex-wrap: wrap; gap: 24px; justify-content: center;'>"

        for _, row in matched.iterrows():
            discounted_price = round(row["price"] * (1 - discount_rate), 2)
            html_gallery += f"""
                <div style='width: 220px; text-align: center; font-family: "Syne", sans-serif;'>
                    <img src="{row['filename']}" style="width: 100%; border-radius: 8px;" />
                    <div style="margin-top: 6px;">
                        <a href="{row['url']}" target="_blank" style="text-decoration: none; font-weight: bold; color: black;">{row['name']}</a><br>
                        <span style='font-size: 0.9rem;'>Weight: {row['weight']} g</span><br>
                        <span class='original-price'>Original Price: {row['price']} €</span><br>
                        <span class='discounted-price'>Now: {discounted_price} € ({int(discount_rate * 100)}% off)</span>
                    </div>
                </div>
            """

        html_gallery += "</div>"
        st.markdown(html_gallery, unsafe_allow_html=True)
    else:
        st.write("No matching designs found.")

