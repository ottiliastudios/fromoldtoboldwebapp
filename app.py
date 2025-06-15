import streamlit as st
st.set_page_config(page_title="From Old to Bold")

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# ---------- MODEL ----------
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

# ---------- APP DESIGN ----------


st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne&display=swap');

        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif !important;
            background-color: #ffffff;
            color: #000000;
        }

        .description-text {
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
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
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """<div style='text-align: center;'>
        <img src='logo.png' width='180'>
    </div>""",
    unsafe_allow_html=True
)

st.markdown('<div class="description-text">Upload a photo of your old piece of jewelry. Our AI estimates the weight and suggests matching new designs!</div>', unsafe_allow_html=True)

st.markdown("""
<div class="external-button-small">
    <a href="https://eager-transform-667249.framer.app/" target="_blank">WHAT IS FROM OLD TO BOLD</a>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT ----------
material = st.selectbox("Select material", ["Silver", "Gold", "Other"])
if material == "Other":
    custom_material = st.text_input("Please specify the material")
    material = custom_material

uploaded_file = st.file_uploader("Upload an image of your old jewelry", type=["jpg", "jpeg", "png"])

# ---------- PREDICT ----------
def predict_weight(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image).item()
    return round(prediction, 2)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    weight = predict_weight(image)
    st.write(f"**Estimated weight:** {weight:.2f} grams")

    df = pd.read_csv("designs.csv", sep=";")
    tolerance = 1.0
    matched = df[
        (abs(df["weight"] - weight) <= tolerance) &
        (df["material"].str.lower() == material.lower())
    ]

        st.subheader("Matching designs:")
if not matched.empty:
    img_paths = matched["filename"].tolist()
    captions = matched.apply(
        lambda row: f"<a href='{row['url']}' target='_blank'>{row['name']} – {row['weight']} g</a>",
        axis=1
    ).tolist()

    idx = st.slider("Browse matching designs", 0, len(img_paths) - 1, 0)
    st.image(img_paths[idx], width=300)
    st.markdown(f"<div style='text-align: center;'>{captions[idx]}</div>", unsafe_allow_html=True)
else:
    st.write("No matching designs found.")

