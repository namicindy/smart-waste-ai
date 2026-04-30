import torch
import torch.nn as nn
from torch.optim import Adam
from model import get_model
from utils import get_dataloaders

# ⚙️ Paramètres
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_DIR = "data"

# 📦 Charger les données
train_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

# 🤖 Charger le modèle
model = get_model(num_classes=len(classes))

# 📉 Loss function + Optimizer
criterion = nn.CrossEntropyLoss()  # mesure les erreurs
optimizer = Adam(model.fc.parameters(), lr=LEARNING_RATE)  # corrige les erreurs

# 🏋️ Boucle d'entraînement
print("\n🚀 Entraînement démarré !\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()          # remettre les gradients à zéro
        outputs = model(images)        # prédiction
        loss = criterion(outputs, labels)  # calcul de l'erreur
        loss.backward()               # corriger
        optimizer.step()              # mettre à jour

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} → Loss: {total_loss:.2f} | Accuracy: {accuracy:.1f}%")

# 💾 Sauvegarder le modèle
torch.save(model.state_dict(), "model.pth")
print("\n✅ Modèle sauvegardé dans model.pth !")