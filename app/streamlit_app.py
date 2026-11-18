"""
AgroExport Ica - Estimador de Cosecha (Streamlit)
==================================================
Interfaz gráfica para agrónomos en campo.
Permite estimar la producción de uva de mesa ingresando
las variables del lote mediante sliders interactivos.
"""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Configuración de paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroExport Ica - Estimador de Cosecha",
    page_icon="🍇",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Funciones de carga de modelos (con cache)
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_modelos():
    """Carga los modelos entrenados (cacheados por Streamlit)."""
    modelos = {}

    # Red neuronal
    keras_path = MODELS_DIR / "modelo_red_neuronal.keras"
    if not keras_path.exists():
        keras_path = MODELS_DIR / "modelo_red_neuronal.h5"
    if keras_path.exists():
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            modelos["red_neuronal"] = tf.keras.models.load_model(str(keras_path))
        except Exception as e:
            st.warning(f"No se pudo cargar red neuronal: {e}")

    # Scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        try:
            import joblib
            modelos["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"No se pudo cargar scaler: {e}")

    # Modelo lineal múltiple (fallback)
    lineal_path = MODELS_DIR / "modelo_lineal_multiple.joblib"
    if lineal_path.exists():
        try:
            import joblib
            modelos["lineal_multiple"] = joblib.load(lineal_path)
        except Exception as e:
            st.warning(f"No se pudo cargar modelo lineal: {e}")

    return modelos


def predecir(modelos, hectareas, litros_riego, radiacion, kg_fert, antiguedad):
    """Realiza la predicción con el mejor modelo disponible."""
    features = np.array([[hectareas, litros_riego, radiacion, kg_fert, antiguedad]])

    modelo_usado = None
    prediccion = None

    if "red_neuronal" in modelos and "scaler" in modelos:
        try:
            features_scaled = modelos["scaler"].transform(features)
            prediccion = float(modelos["red_neuronal"].predict(features_scaled, verbose=0)[0][0])
            modelo_usado = "Red Neuronal"
        except Exception:
            pass

    if prediccion is None and "lineal_multiple" in modelos and "scaler" in modelos:
        try:
            prediccion = float(modelos["lineal_multiple"].predict(features)[0])
            modelo_usado = "Regresión Lineal Múltiple"
        except Exception:
            pass

    if prediccion is not None:
        prediccion = max(0.0, prediccion)

    return prediccion, modelo_usado


# ---------------------------------------------------------------------------
# Objetivos del escenario semi-sintético.
# ---------------------------------------------------------------------------
RENDIMIENTO_PROMEDIO_HA = 22.0  # objetivo sintético, no promedio histórico oficial
RENDIMIENTO_MIN_HA = 20.0
RENDIMIENTO_MAX_HA = 25.0

# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title("🍇 AgroExport Ica - Estimador de Cosecha")
st.markdown(
    "**Herramienta de apoyo para agrónomos** · Estima la producción de uva de mesa "
    "en la región Ica, Perú, utilizando modelos de Machine Learning."
)

st.divider()

# Sidebar - Parámetros de entrada
with st.sidebar:
    st.header("📊 Parámetros del Lote")
    st.markdown("Ajuste los valores según las condiciones del lote:")

    hectareas = st.slider(
        "Hectáreas del lote",
        min_value=2.0,
        max_value=50.0,
        value=15.0,
        step=0.5,
        help="Superficie total del lote en hectáreas",
    )

    litros_riego_ha = st.slider(
        "Riego (m³/ha/campaña)",
        min_value=5000,
        max_value=12000,
        value=8500,
        step=100,
        help="Rango supuesto para los datos semi-sintéticos: 5,000-12,000 m³/ha",
    )
    litros_riego_total = float(litros_riego_ha * 1000 * hectareas)

    radiacion = st.slider(
        "Radiación solar acumulada (kWh/m²)",
        min_value=1238.3,
        max_value=1340.4,
        value=1263.5,
        step=0.1,
        help="Radiación NASA POWER observada por campaña; el modelo fue entrenado en este rango.",
    )

    kg_fert_ha = st.slider(
        "Fertilizante (kg/ha)",
        min_value=150,
        max_value=600,
        value=350,
        step=10,
        help="Kilogramos de fertilizante aplicado por hectárea",
    )
    kg_fert_total = float(kg_fert_ha * hectareas)

    antiguedad = st.slider(
        "Antigüedad del viñedo (años)",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Años desde la plantación del viñedo",
    )

    st.divider()
    st.caption(
        "La radiación procede de NASA POWER. Las cosechas e insumos por lote son "
        "semi-sintéticos; MIDAGRI aporta estadísticas agregadas. Ver data/FUENTES.md."
    )

# Cargar modelos
modelos = cargar_modelos()

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    # Botón de predicción
    if st.button("🔮 Estimar Cosecha", type="primary", use_container_width=True):
        if not modelos:
            st.error(
                "⚠️ No se encontraron modelos entrenados. "
                "Ejecuta primero `python src/modelos.py` para entrenar los modelos."
            )
        else:
            prediccion, modelo_usado = predecir(
                modelos, hectareas, litros_riego_total, radiacion, kg_fert_total, antiguedad
            )

            if prediccion is not None:
                st.session_state["ultima_prediccion"] = prediccion
                st.session_state["ultimo_modelo"] = modelo_usado
                st.session_state["ultimas_hectareas"] = hectareas
            else:
                st.error("No se pudo realizar la predicción.")

# Mostrar resultados si hay predicción
if "ultima_prediccion" in st.session_state:
    prediccion = st.session_state["ultima_prediccion"]
    modelo_usado = st.session_state["ultimo_modelo"]
    hectareas_pred = st.session_state["ultimas_hectareas"]

    rendimiento_estimado = prediccion / hectareas_pred
    referencia_prototipo = RENDIMIENTO_PROMEDIO_HA * hectareas_pred
    delta_toneladas = prediccion - referencia_prototipo
    delta_pct = ((prediccion / referencia_prototipo) - 1) * 100 if referencia_prototipo > 0 else 0

    st.divider()
    st.subheader("📈 Resultado de la Estimación")

    # Métricas principales
    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.metric(
            label="🍇 Toneladas Estimadas",
            value=f"{prediccion:,.1f} t",
            delta=f"{delta_toneladas:+.1f} t vs objetivo",
        )

    with mcol2:
        st.metric(
            label="📐 Rendimiento por Hectárea",
            value=f"{rendimiento_estimado:,.1f} t/ha",
            delta=f"{rendimiento_estimado - RENDIMIENTO_PROMEDIO_HA:+.1f} t/ha vs objetivo",
        )

    with mcol3:
        # Confianza
        if 15 <= rendimiento_estimado <= 30:
            confianza = "🟢 Alta"
        elif 10 <= rendimiento_estimado <= 40:
            confianza = "🟡 Media"
        else:
            confianza = "🔴 Baja"

        st.metric(
            label="🎯 Confianza",
            value=confianza,
        )
        st.caption(f"Modelo: {modelo_usado}")

    # Comparación con el objetivo sintético.
    st.divider()
    st.subheader("📊 Comparación con Objetivo del Prototipo")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Datos para el gráfico
    categorias = ["Objetivo Sintético\n(22 t/ha)", "Tu Estimación"]
    valores = [RENDIMIENTO_PROMEDIO_HA, rendimiento_estimado]
    colores = ["#6c757d", "#28a745" if rendimiento_estimado >= RENDIMIENTO_PROMEDIO_HA else "#dc3545"]

    bars = ax.barh(categorias, valores, color=colores, height=0.5, edgecolor="white", linewidth=1.5)

    # Rango de diseño sintético, no intervalo histórico observado.
    ax.axvspan(RENDIMIENTO_MIN_HA, RENDIMIENTO_MAX_HA, alpha=0.1, color="#007bff", label="Rango del escenario sintético (20-25 t/ha)")

    # Etiquetas en las barras
    for bar, val in zip(bars, valores):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} t/ha",
            va="center", fontweight="bold", fontsize=12,
        )

    ax.set_xlabel("Rendimiento (t/ha)", fontsize=11)
    ax.set_title("Rendimiento Estimado vs Objetivo del Prototipo", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, max(valores) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    # Indicador gauge-style simplificado
    st.divider()
    st.subheader("🎯 Indicador de Rendimiento")

    fig2, ax2 = plt.subplots(figsize=(10, 2.5))

    # Barra de fondo con zonas de color
    zone_width = 50
    ax2.barh(0, 10, left=0, color="#dc3545", alpha=0.3, height=0.6)   # Bajo: 0-10
    ax2.barh(0, 10, left=10, color="#ffc107", alpha=0.3, height=0.6)  # Medio: 10-20
    ax2.barh(0, 10, left=20, color="#28a745", alpha=0.3, height=0.6)  # Óptimo: 20-30
    ax2.barh(0, 10, left=30, color="#ffc107", alpha=0.3, height=0.6)  # Alto: 30-40
    ax2.barh(0, 10, left=40, color="#dc3545", alpha=0.3, height=0.6)  # Excesivo: 40-50

    # Marcador de la estimación
    marker_pos = min(max(rendimiento_estimado, 0), 50)
    ax2.plot(marker_pos, 0, marker="v", markersize=20, color="#212529", zorder=5)
    ax2.annotate(
        f"{rendimiento_estimado:.1f} t/ha",
        xy=(marker_pos, 0), xytext=(marker_pos, -0.5),
        fontsize=12, fontweight="bold", ha="center",
    )

    # Marcador del objetivo sintético
    ax2.axvline(RENDIMIENTO_PROMEDIO_HA, color="#007bff", linewidth=2, linestyle="--", zorder=4)
    ax2.annotate(
        f"Objetivo: {RENDIMIENTO_PROMEDIO_HA} t/ha",
        xy=(RENDIMIENTO_PROMEDIO_HA, 0), xytext=(RENDIMIENTO_PROMEDIO_HA, 0.5),
        fontsize=9, ha="center", color="#007bff",
    )

    # Etiquetas de zonas
    for x, label in [(5, "Bajo"), (15, "Medio"), (25, "Óptimo"), (35, "Alto"), (45, "Excesivo")]:
        ax2.text(x, 0, label, ha="center", va="center", fontsize=8, alpha=0.6)

    ax2.set_xlim(0, 50)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("Rendimiento (t/ha)")
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("Indicador de Rendimiento por Hectárea", fontsize=11, fontweight="bold")
    plt.tight_layout()

    st.pyplot(fig2)
    plt.close()

# ---------------------------------------------------------------------------
# Acerca del modelo
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Acerca del Modelo", expanded=False):
    st.markdown("""
    ### Modelo de Estimación de Cosecha

    Este prototipo usa una tabla de lotes semi-sintéticos. Las fuentes y sus límites
    están documentados en `data/FUENTES.md`.

    | Fuente | Dato | URL |
    |--------|------|-----|
    | **MIDAGRI/SIEA** | Producción, superficie y rendimiento anual de Ica (2016-2023); no son datos por lote | [Compendio y anuarios](https://www.gob.pe/institucion/midagri/informes-publicaciones/2730325-compendio-anual-de-produccion-agricola) |
    | **FAOSTAT** | No descargado: el API respondió HTTP 521; no se usó | [FAOSTAT QCL](https://www.fao.org/faostat/en/#data/QCL) |
    | **NASA POWER** | Radiación diaria observada, agregada por campaña | [API diaria](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude=-75.73&latitude=-14.07&start=20150101&end=20241231&format=JSON) |

    #### Variables de Entrada
    - **Hectáreas, riego, fertilizante y antigüedad**: rangos simulados para el prototipo.
    - **Radiación solar**: total real NASA POWER de campañas octubre-marzo (kWh/m²).

    #### Modelos Implementados
    1. **Regresión Lineal Simple** (baseline) — solo hectáreas
    2. **Regresión Lineal Múltiple** — todas las variables
    3. **Red Neuronal** (Keras) — captura relaciones no lineales ✅

    #### Efectos No Lineales Capturados
    - 🌊 **Óptimo de riego**: rendimiento sube hasta ~8,500 m³/ha, luego baja (encharcamiento)
    - 🌱 **Rendimientos decrecientes de fertilizante**: más fertilizante ayuda pero con retornos decrecientes
    - 📅 **Curva de antigüedad**: sube años 1-8, meseta 8-18, baja >18 (viñedo envejecido)
    - ☀️ **Saturación de radiación**: efecto positivo pero con saturación

    > **Nota**: Este es un modelo estimativo de apoyo a la decisión.
    > Las predicciones deben complementarse con la experiencia del agrónomo
    > y las condiciones específicas del campo.
    """)

# Footer
st.divider()
st.caption(
    "🍇 AgroExport Ica v1.0 · Radiación NASA POWER y lotes semi-sintéticos; ver data/FUENTES.md · "
    "Región Ica, Perú"
)
