import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import get_model

# Configuration de la page
st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="centered"
)

# Style CSS
st.markdown("""
    <style>
    .main { background-color: red; }
    
    .element.style {
        background-color: violet;
    }
    .title { 
        color: #2e7d32; 
        font-size: 3em; 
        font-weight: bold; 
        text-align: center;
    }
    .subtitle {
        color: #555;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: black;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Conseils de recyclage
CONSEILS = {
    "cardboard": ("♻️", "Carton",    "Poubelle jaune", "#ba68c8"),
    "glass":     ("🫙", "Verre",     "Conteneur à verre vert", "#ba68c8 "),
    "metal":     ("🥫", "Métal",     "Poubelle jaune", "#ba68c8"),
    "paper":     ("📄", "Papier",    "Poubelle jaune", "#ba68c8"),
    "plastic":   ("🧴", "Plastique", "Poubelle jaune", "#ba68c8"),
    "trash":     ("🗑️", "Déchet",   "Poubelle noire", "#ba68c8"),
}

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Charger le modèle
@st.cache_resource
def load_model():
    model = get_model(num_classes=6)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

# Préparer l'image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Afficher le résultat
def afficher_resultat(image, classe, confiance):
    emoji, nom, conseil, couleur = CONSEILS[classe]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Image analysée", use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
        <div class="result-box" style="background-color:{couleur}">
            <h2>{emoji} {nom.upper()}</h2>
            <h3>🎯 Confiance : {confiance:.1f}%</h3>
            <h3>🗑️ Où le jeter : <b>{conseil}</b></h3>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("####")
    st.progress(int(confiance))

# Titre
st.markdown('<p class="title">🌍 Smart Waste AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Montre un déchet et je te dis où le jeter !</p>', unsafe_allow_html=True)

model = load_model()

# Deux onglets : Upload et Caméra
tab1, tab2 = st.tabs(["📁 Upload une image", "📷 Utiliser la caméra"])

# --- Onglet 1 : Upload ---
with tab1:
    uploaded_file = st.file_uploader("Upload une image de déchet", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        tensor = preprocess(image)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

        classe = CLASSES[predicted.item()]
        confiance = confidence.item() * 100
        afficher_resultat(image, classe, confiance)

# --- Onglet 2 : Caméra ---
with tab2:
    st.markdown("### 📷 Montre ton déchet à la caméra")
    camera_photo = st.camera_input("Prends une photo !")

    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")
        tensor = preprocess(image)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

        classe = CLASSES[predicted.item()]
        confiance = confidence.item() * 100
        afficher_resultat(image, classe, confiance)