# test des versions des librairies pour éviter les problèmes de compatibilité
import torch
import torchvision
import streamlit

print(" torch :", torch.__version__)
print(" torchvision :", torchvision.__version__)
print(" streamlit :", streamlit.__version__)
print(" Tout est prêt !")