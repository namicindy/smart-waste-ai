from model import get_model

model = get_model()
print(" Modèle prêt !")
print(f"   → Dernière couche : {model.fc}")