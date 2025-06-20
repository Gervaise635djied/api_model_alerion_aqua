
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import traceback
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
    return {"message": "Bienvenue sur l’API de prédiction d’espèces aquacoles (modèle Random Forest)"}

# === Route de prédiction ===
@app.post("/predict")
def predict_species(data: AquacultureInput):
    try:
        input_array = np.array([[data.temperature, data.ph, data.nh3, data.oxygen, data.salinite]])

        # Prédiction
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

