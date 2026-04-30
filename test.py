import torch
from model import get_model
from utils import get_dataloaders

# 📦 Charger les données
_, test_loader, classes = get_dataloaders("data")

# 🤖 Charger le modèle sauvegardé
model = get_model(num_classes=len(classes))
model.load_state_dict(torch.load("model.pth"))
model.eval()  # mode évaluation (pas d'entraînement)

# 📊 Tester sur les images de test
correct = 0
total = 0

with torch.no_grad():  # pas besoin de calculer les gradients
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Accuracy sur les images de test : {accuracy:.1f}%")