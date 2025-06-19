
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib  # pour charger le LabelEncoder

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Autoriser toutes les origines (sécurise plus tard si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# === Définir les données attendues pour la prédiction ===
class AquacultureInput(BaseModel):
    temperature: float
    ph: float
    nh3: float
    oxygen: float
    salinite: float

# === Charger le modèle ===
def load_model():
    return joblib.load("aquaculture_model.pkl")

# 🔹 Charger le LabelEncoder
def load_encoder():
    return joblib.load("label_encoder.pkl")

# Charger au démarrage
model = load_model()
label_encoder = load_encoder()

# === Routes === test de connexion
@app.get("/")
def great():
    return {"message":"Bienvenue sur l’API de prédiction d’espèces aquacoles (modèle Random Forest)"}

@app.post("/predict")
def predict_species(data: AquacultureInput):
    # Convertir les données en tableau numpy
    input_array = np.array([[data.temperature, data.ph,data.nh3, data.oxygen, data.salinite]])

    # Faire la prédiction
    predicted_index = int(model.predict(input_array)[0])


    # Utiliser le label encoder pour obtenir le nom de l'espèce
    predicted_species = label_encoder.inverse_transform([predicted_index])[0]

    return {
        "predicted_class_index": int(predicted_index),
        "predicted_species": predicted_species
    }







