
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import numpy as np
import joblib
import traceback
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Récupération de la clé depuis les variables d'environnement
API_KEY = os.environ.get("API_KEY")

# Solution pour le développement local
if not API_KEY:
    import secrets
    API_KEY = secrets.token_urlsafe(32)
    print(f"Clé API temporaire : {API_KEY}")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide ou manquante",
            headers={"WWW-Authenticate": "APIKey"}
        )
    return api_key

# === Modèle de données ===
class AquacultureInput(BaseModel):
    temperature: float
    ph: float
    nh3: float
    oxygen: float
    salinite: float

# === Chargement du modèle et du label encoder ===
try:
    model = joblib.load("aquaculture_model.pkl")
except Exception as e:
    raise RuntimeError("Erreur lors du chargement du modèle : " + str(e))

try:
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    raise RuntimeError("Erreur lors du chargement du LabelEncoder : " + str(e))

# === Gestion des erreurs globales ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du serveur",
            "error": str(exc),
            "trace": traceback.format_exc()
        },
    )

# === Route d'accueil ===
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction d'espèces aquacoles"}

# === Route de prédiction PROTÉGÉE ===
@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_species(data: AquacultureInput):
    try:
        input_array = np.array([[data.temperature, data.ph, data.nh3, data.oxygen, data.salinite]])
        predicted_index = int(model.predict(input_array)[0])
        predicted_species = label_encoder.inverse_transform([predicted_index])[0]

        return {
            "predicted_class_index": predicted_index,
            "predicted_species": predicted_species
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Erreur de valeur : {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8001)

