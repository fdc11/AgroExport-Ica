"""
AgroExport Ica - API de Predicción de Cosecha
==============================================
API REST con FastAPI para estimar toneladas de uva de mesa
usando modelos de Machine Learning entrenados con datos de Ica, Perú.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Agregar src/ al path para poder importar utilidades si se necesitan
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionInput(BaseModel):
    """Datos de entrada para la predicción de cosecha."""
    hectareas: float = Field(..., gt=0, le=200, description="Superficie del lote en hectáreas (2-50 típico)")
    litros_riego: float = Field(..., gt=0, description="Litros totales de riego del lote por campaña")
    radiacion_solar_acumulada: float = Field(
        ..., ge=1238.3, le=1340.4,
        description="Radiación acumulada en el rango observado por NASA POWER, kWh/m²/campaña",
    )
    kg_fertilizante: float = Field(..., gt=0, description="Kilogramos totales de fertilizante aplicado")
    antiguedad_terreno: float = Field(..., ge=1, le=30, description="Antigüedad simulada del viñedo (1-30 años)")

    @model_validator(mode="after")
    def validar_rangos_del_modelo(self):
        riego_m3_ha = self.litros_riego / self.hectareas / 1000
        fertilizante_kg_ha = self.kg_fertilizante / self.hectareas
        if not 5000 <= riego_m3_ha <= 12000:
            raise ValueError("El riego debe estar entre 5,000 y 12,000 m³/ha/campaña")
        if not 150 <= fertilizante_kg_ha <= 600:
            raise ValueError("El fertilizante debe estar entre 150 y 600 kg/ha")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hectareas": 15.0,
                    "litros_riego": 120000000.0,
                    "radiacion_solar_acumulada": 1125.0,
                    "kg_fertilizante": 4500.0,
                    "antiguedad_terreno": 10.0,
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Resultado de la predicción de cosecha."""
    toneladas_estimadas: float
    modelo_usado: str
    confianza: str


class HealthResponse(BaseModel):
    status: str
    modelos_cargados: bool
    version: str


# ---------------------------------------------------------------------------
# Global model holders
# ---------------------------------------------------------------------------
_modelos: dict = {}


def _cargar_modelos():
    """Carga los modelos entrenados desde disco."""
    global _modelos

    # Red neuronal (Keras)
    keras_path = MODELS_DIR / "modelo_red_neuronal.keras"
    if not keras_path.exists():
        # Intentar también .h5 por compatibilidad
        keras_path = MODELS_DIR / "modelo_red_neuronal.h5"

    if keras_path.exists():
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            _modelos["red_neuronal"] = tf.keras.models.load_model(str(keras_path))
            print(f"✅ Red neuronal cargada desde {keras_path.name}")
        except Exception as e:
            print(f"⚠️  No se pudo cargar red neuronal: {e}")

    # Scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        try:
            import joblib
            _modelos["scaler"] = joblib.load(scaler_path)
            print("✅ Scaler cargado")
        except Exception as e:
            print(f"⚠️  No se pudo cargar scaler: {e}")

    # Modelo lineal múltiple
    lineal_path = MODELS_DIR / "modelo_lineal_multiple.joblib"
    if lineal_path.exists():
        try:
            import joblib
            _modelos["lineal_multiple"] = joblib.load(lineal_path)
            print("✅ Modelo lineal múltiple cargado")
        except Exception as e:
            print(f"⚠️  No se pudo cargar modelo lineal: {e}")

    if not _modelos:
        print("⚠️  No se encontraron modelos en", MODELS_DIR)
        print("   Ejecuta primero: python src/modelos.py")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelos al iniciar la API."""
    print("🚀 Iniciando AgroExport Ica API...")
    _cargar_modelos()
    yield
    print("👋 Cerrando AgroExport Ica API...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgroExport Ica - API de Predicción de Cosecha",
    description=(
        "API REST para estimar la producción de uva de mesa en la región Ica, Perú. "
        "Prototipo entrenado con lotes semi-sintéticos y radiación NASA POWER; "
        "MIDAGRI aporta estadísticas agregadas, no datos por lote. FAOSTAT no se descargó."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS habilitado para cualquier origen (desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _determinar_confianza(toneladas: float, hectareas: float) -> str:
    """
    Determina la confianza de la predicción basándose en si el rendimiento
    por hectárea cae dentro de rangos agronómicamente esperados.
    """
    rendimiento_ha = toneladas / hectareas if hectareas > 0 else 0

    # Umbrales de orientación del prototipo; no son intervalos oficiales observados.
    if 15 <= rendimiento_ha <= 30:
        return "alta"
    elif 10 <= rendimiento_ha <= 40:
        return "media"
    else:
        return "baja"


def _predecir(datos: PredictionInput) -> PredictionOutput:
    """Realiza la predicción usando el mejor modelo disponible."""
    features = np.array([[
        datos.hectareas,
        datos.litros_riego,
        datos.radiacion_solar_acumulada,
        datos.kg_fertilizante,
        datos.antiguedad_terreno,
    ]])

    modelo_usado = "ninguno"
    prediccion = None

    # Prioridad: red neuronal > lineal múltiple
    if "red_neuronal" in _modelos and "scaler" in _modelos:
        try:
            features_scaled = _modelos["scaler"].transform(features)
            prediccion = float(_modelos["red_neuronal"].predict(features_scaled, verbose=0)[0][0])
            modelo_usado = "red_neuronal"
        except Exception:
            pass

    if prediccion is None and "lineal_multiple" in _modelos and "scaler" in _modelos:
        try:
            prediccion = float(_modelos["lineal_multiple"].predict(features)[0])
            modelo_usado = "lineal_multiple"
        except Exception:
            pass

    if prediccion is None:
        raise HTTPException(
            status_code=503,
            detail="No hay modelos disponibles. Ejecuta primero 'python src/modelos.py'.",
        )

    # Asegurar que la predicción sea positiva
    prediccion = max(0.0, prediccion)

    confianza = _determinar_confianza(prediccion, datos.hectareas)

    return PredictionOutput(
        toneladas_estimadas=round(prediccion, 2),
        modelo_usado=modelo_usado,
        confianza=confianza,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """Información general de la API."""
    return {
        "nombre": "AgroExport Ica - API de Predicción de Cosecha",
        "version": "1.0.0",
        "descripcion": (
            "Estima la producción de uva de mesa en Ica usando modelos de ML. "
            "Envía un POST a /predict con los datos del lote."
        ),
        "endpoints": {
            "GET /": "Esta información",
            "GET /health": "Estado del servicio",
            "GET /docs": "Documentación interactiva (Swagger UI)",
            "POST /predict": "Predicción de cosecha",
        },
        "modelos_disponibles": list(
            k for k in _modelos if k != "scaler"
        ),
        "ejemplo_input": {
            "hectareas": 15.0,
            "litros_riego": 120000000.0,
            "radiacion_solar_acumulada": 1125.0,
            "kg_fertilizante": 4500.0,
            "antiguedad_terreno": 10.0,
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Verifica el estado del servicio y los modelos."""
    modelos_ok = len(_modelos) > 0
    return HealthResponse(
        status="ok" if modelos_ok else "sin_modelos",
        modelos_cargados=modelos_ok,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predicción"])
def predict(datos: PredictionInput):
    """
    Estima las toneladas de uva de mesa para un lote en Ica.

    Recibe las 5 variables de entrada del lote y retorna la estimación
    de producción, el modelo utilizado y el nivel de confianza.
    """
    return _predecir(datos)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
