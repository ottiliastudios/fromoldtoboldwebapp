import streamlit as st
# ---------- Seiteneinstellungen ----------
st.set_page_config(page_title="From Old to Bold")
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image



# ---------- Modell ----------
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

# ---------- Styles ----------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne&display=swap');
        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif !important;
        }

        .description-text {
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-family: 'Syne', sans-serif !important;
        }

        .external-button-small {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
            font-family: 'Syne', sans-serif !important;
        }

        .external-button-small a {
            background-color: black;
            color: white;
            padding: 6px 12px;
            font-size: 0.85rem;
            border-radius: 6px;
            text-decoration: none;
            font-family: 'Syne', sans-serif !important;
        }

        .original-price {
            text-decoration: line-through;
            color: gray;
            font-family: 'Syne', sans-serif !important;
        }

        .discounted-price {
            color: green;
            font-weight: bold;
            font-family: 'Syne', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Logo ----------
cols = st.columns([1, 1, 1])
with cols[1]:
    st.image("logo.png", width=180)

# ---------- Intro Text + Link ----------
st.markdown('<div class="description-text">Upload a photo of your old piece of jewelry next to a ruler. Our AI estimates the weight and suggests matching new designs!</div>', unsafe_allow_html=True)

st.markdown("""
<div class="external-button-small">
    <a href="https://eager-transform-667249.framer.app/" target="_blank">WHAT IS FROM OLD TO BOLD</a>
</div>
""", unsafe_allow_html=True)

# ---------- Auswahl + Upload ----------
material = st.selectbox("Select material", ["Silver", "Gold", "Other"])
if material == "Other":
    custom_material = st.text_input("Please specify the material")
    material = custom_material

uploaded_file = st.file_uploader("Upload an image of your old jewelry", type=["jpg", "jpeg", "png"])

# ---------- Gewichtsschätzung & Vorschläge ----------
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    img_col = st.columns([1, 1, 1])[1]
    with img_col:
        st.image(image, caption="Uploaded image", width=200)
        weight = predict_weight(image)
        st.markdown(f"<p style='text-align: center; font-weight: bold;'>Estimated weight: {weight:.2f} grams</p>", unsafe_allow_html=True)

    # Vorschläge anzeigen
    df = pd.read_csv("designs.csv", sep=";")
    tolerance = 1.0
    matched = df[
        (abs(df["weight"] - weight) <= tolerance) &
        (df["material"].str.lower() == material.lower())
    ]

    if not matched.empty:
        st.markdown("<h4 style='margin-left: 16px;'>Matching Designs:</h4>", unsafe_allow_html=True)

        rows = [matched.iloc[i:i+3] for i in range(0, len(matched), 3)]

        for row_group in rows:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(row_group.iterrows()):
                with cols[idx]:
                    st.image(row["filename"], use_container_width=True)
                    st.markdown(
                    f"""
        <div style='text-align: center; margin-top: -8px; font-family: "Syne", sans-serif !important;'>
        <a href='{row['url']}' target='_blank' style='text-decoration: none; font-weight: bold;'>{row['name']}</a><br>
        <span style='font-size: 0.9rem; font-family: 'Syne', sans-serif !important;'>Weight: {row['weight']} g</span><br>
        <span class='original-price'>Original Price: {row['price']} €</span><br>
        <span class='discounted-price'>Now: {round(row['price'] * 0.9, 2)} € (10% off)</span>
    </div>
    """,
    unsafe_allow_html=True
)

    else:
        st.write("No matching designs found.")
