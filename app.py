# --------------------------------------------------------------
# Streamlit Deployment for PyTorch L2-Regularized Model
# --------------------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------------
# 1️⃣ Load Saved Model
# --------------------------------------------------------------
class YeastModelL2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(YeastModelL2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# Model parameters
INPUT_DIM = 8          # Number of features
HIDDEN_DIM = 32        # Hidden layer size
OUTPUT_DIM = 10        # Number of target classes
MODEL_PATH = "yeast_model_l2.pth"

# Load model
model = YeastModelL2(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --------------------------------------------------------------
# 2️⃣ Sidebar: Feature & Target Info
# --------------------------------------------------------------
st.sidebar.title("Dataset Info")

# Feature names (replace with your actual feature names)
feature_names = [
    "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"
]

# Target classes with full description
target_info = {
    "CYT": "Cytoplasm: general metabolic and regulatory functions.",
    "NUC": "Nucleus: contains DNA, transcription, and RNA processing.",
    "MIT": "Mitochondrion: energy production, respiration.",
    "ME3": "Membrane protein, class 3: involved in transport across membranes.",
    "ME2": "Membrane protein, class 2: involved in structural functions.",
    "ME1": "Membrane protein, class 1: general membrane-associated activity.",
    "EXC": "Extracellular: secreted proteins outside the cell.",
    "VAC": "Vacuole: storage and degradation of cellular materials.",
    "POX": "Peroxisome: fatty acid oxidation and reactive oxygen metabolism.",
    "ERL": "Endoplasmic reticulum lumen: protein folding and processing."
}

# Display features in sidebar
st.sidebar.subheader("Features")
for f in feature_names:
    st.sidebar.write(f"- {f}")

# Display target classes with description
st.sidebar.subheader("Target Classes & Properties")
for cls, desc in target_info.items():
    st.sidebar.write(f"- **{cls}**: {desc}")

# Sidebar explanation of target variable
st.sidebar.subheader("Target Variable Meaning")
st.sidebar.write(
    "Yeast localization sites: each class represents a different cellular compartment where the protein is located."
)

# --------------------------------------------------------------
# 3️⃣ Streamlit App UI: Inputs
# --------------------------------------------------------------
st.title("Yeast Classification Predictor (L2 Regularized Model)")
st.markdown("Enter feature values to predict the yeast class:")

input_features = []
for i, name in enumerate(feature_names):
    val = st.number_input(f"{name}", value=0.0, format="%.4f")
    input_features.append(val)

# Convert input to tensor
input_tensor = torch.tensor([input_features], dtype=torch.float32)

# --------------------------------------------------------------
# 4️⃣ Predict Button
# --------------------------------------------------------------
if st.button("Predict"):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()
        pred_class_name = list(target_info.keys())[pred_class_idx]
        pred_class_desc = target_info[pred_class_name]

    st.success(f"Predicted Class: {pred_class_name}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.write(f"**Class Meaning:** {pred_class_desc}")

# --------------------------------------------------------------
# 5️⃣ Optional: Show Probabilities
# --------------------------------------------------------------
if st.checkbox("Show Probabilities for All Classes"):
    class_probs = probs.numpy()[0]
    for i, cls in enumerate(target_info.keys()):
        st.write(f"**{cls}**: {class_probs[i]*100:.2f}%")
