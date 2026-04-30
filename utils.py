from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Transformation des images pour le modèle
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),      # Redimensionnement en 224x224
        transforms.RandomHorizontalFlip(),  # Retourner aléatoirement (data augmentation)
        transforms.ToTensor(),              # Convertir en tensor
        transforms.Normalize(              # Normaliser les couleurs
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Chargement du dataset
def get_dataloaders(data_dir, batch_size=32):
    dataset = datasets.ImageFolder(data_dir, transform=get_transforms())

    # Découpaage : 80% entraînement, 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Création des dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f" Dataset chargé : {len(dataset)} images")
    print(f"   → Entraînement : {train_size} images")
    print(f"   → Test : {test_size} images")
    print(f"   → Classes : {dataset.classes}")

    return train_loader, test_loader, dataset.classes