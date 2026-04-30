import torch.nn as nn
from torchvision import models

def get_model(num_classes=6):
    # Charger ResNet18 déjà pré-entraîné
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Geler toutes les couches (on ne retouche pas ce qu'il sait déjà)
    for param in model.parameters():
        param.requires_grad = False

    # Remplacer la dernière couche pour NOS 6 catégories de déchets
    model.fc = nn.Linear(512, num_classes)

    return model
