# %% [markdown]
# # 01 - Construcción del dataset de cosechas de Ica
#
# La radiación procede de NASA POWER y las estadísticas anuales regionales de
# uva de MIDAGRI/SIEA. Los lotes y sus cosechas son semi-sintéticos, no datos
# históricos de AgroExport. FAOSTAT no se descargó (HTTP 521); ver `data/FUENTES.md`.

# %%
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from generar_dataset import generar_dataset

print(f"Directorio del proyecto: {PROJECT_ROOT}")
print(f"Python: {sys.version}")

# %% [markdown]
# ## Generación y calibración oficial
#
# El generador requiere los archivos NASA POWER y los anuarios MIDAGRI 2016-2023
# en `data/raw/`. No usa sustitutos ni CSVs anteriores si falta alguna fuente.
# Informa por separado el rendimiento anual oficial y el objetivo semi-sintético.

# %%
df = generar_dataset()
serie_midagri = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "midagri_uva_ica_2016_2023.csv"
)

print("\nSerie anual MIDAGRI/SIEA de uva para Ica:")
print(
    serie_midagri[
        [
            "anio",
            "produccion_t",
            "superficie_cosechada_ha",
            "rendimiento_calculado_t_ha",
        ]
    ].to_string(index=False)
)
rendimiento_oficial = serie_midagri["rendimiento_calculado_t_ha"]
print(
    "\nRendimiento anual MIDAGRI 2016-2023: "
    f"{rendimiento_oficial.mean():.2f} ± {rendimiento_oficial.std(ddof=1):.2f} t/ha "
    "(media simple ± desviación estándar muestral)"
)

# %% [markdown]
# ## Verificación del dataset semi-sintético
#
# La columna `campania` permite rastrear cada radiación hasta un total real
# octubre-marzo; solo existen nueve campañas completas en el periodo descargado.
# Las métricas calculadas sobre `toneladas` describen datos simulados y no validan
# el rendimiento futuro de la empresa.

# %%
csv_path = PROJECT_ROOT / "data" / "processed" / "historial_cosechas_ica.csv"
df = pd.read_csv(csv_path)
rendimiento_simulado = df["toneladas"] / df["hectareas"]

print(f"Archivo: {csv_path}")
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Valores nulos: {int(df.isna().sum().sum())}")
print(
    "Rendimiento semi-sintético: "
    f"{rendimiento_simulado.mean():.2f} ± {rendimiento_simulado.std():.2f} t/ha"
)
print("\nPrimeras filas:")
print(df.head().to_string(index=False))

# %% [markdown]
# ## Conclusión
#
# El CSV reúne 400 lotes semi-sintéticos y radiación NASA POWER observada,
# agregada por nueve campañas completas. La media objetivo de 22 t/ha es un
# parámetro explícito del escenario y no una estadística oficial. MIDAGRI/SIEA
# aporta el resumen anual de Ica de 2016-2023; FAOSTAT no pudo descargarse.
# No se deben interpretar los resultados de modelos entrenados con este CSV
# como evidencia de precisión de cosecha real hasta incorporar datos de campo.
