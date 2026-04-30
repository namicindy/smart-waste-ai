# 🌍 Smart Waste AI

Une application web qui reconnaît un déchet à partir d'une photo 
et indique où le jeter.

## 🎯 Objectif
Aider au tri sélectif grâce à l'intelligence artificielle.

## 🤖 Technologie
- Modèle : ResNet18 (transfer learning)
- Framework : PyTorch
- Interface : Streamlit
- Dataset : TrashNet (2527 images, 6 classes)

## 📦 Installation
pip install -r requirements.txt

## 🚀 Lancer l'app
streamlit run app.py

## 🗑️ Classes reconnues
- Cardboard (carton)
- Glass (verre)
- Metal (métal)
- Paper (papier)
- Plastic (plastique)
- Trash (ordures)

## 📊 Performance
- Accuracy entraînement : 85.8%
- Accuracy test : 86.4%