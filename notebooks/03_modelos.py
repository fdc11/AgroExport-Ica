# %% [markdown]
# # 03 - Entrenamiento y Comparación de Modelos
#
# **Proyecto**: AgroExport Ica
# **Objetivo**: Entrenar 3 modelos de predicción de cosecha de uva de mesa,
# compararlos con métricas estándar, y recomendar el mejor para producción.
#
# **Modelos**:
# 1. Regresión Lineal Simple (baseline - solo hectáreas)
# 2. Regresión Lineal Múltiple (todas las variables)
# 3. Red Neuronal (Keras - captura no linealidades)

# %% Celda 1: Cargar datos y split
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "historial_cosechas_ica.csv"
GRAFICOS_DIR = PROJECT_ROOT / "graficos"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(f"📁 Proyecto: {PROJECT_ROOT}")
print(f"📄 Datos: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

# Definir features y target
FEATURES = ['hectareas', 'litros_riego', 'radiacion_solar_acumulada', 'kg_fertilizante', 'antiguedad_terreno']
TARGET = 'toneladas'

X = df[FEATURES].values
y = df[TARGET].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Split de datos:")
print(f"   Train: {X_train.shape[0]} muestras")
print(f"   Test:  {X_test.shape[0]} muestras")
print(f"   Features: {FEATURES}")
print(f"   Target: {TARGET}")

# %% Celda 2: Modelo 1 - Regresión Lineal Simple
print("\n" + "=" * 70)
print("📐 MODELO 1: Regresión Lineal Simple (baseline)")
print("=" * 70)
print("   Variable predictora: solo 'hectareas'")
print("   Propósito: establecer una línea base mínima\n")

# Solo usar hectáreas (columna 0)
X_train_simple = X_train[:, 0].reshape(-1, 1)
X_test_simple = X_test[:, 0].reshape(-1, 1)

modelo_lineal_simple = LinearRegression()
modelo_lineal_simple.fit(X_train_simple, y_train)

y_pred_simple = modelo_lineal_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)

print(f"   Coeficiente (hectáreas): {modelo_lineal_simple.coef_[0]:.4f}")
print(f"   Intercepto: {modelo_lineal_simple.intercept_:.4f}")
print(f"\n   📏 Métricas en Test:")
print(f"      MSE:  {mse_simple:.2f}")
print(f"      RMSE: {rmse_simple:.2f} toneladas")
print(f"      MAE:  {mae_simple:.2f} toneladas")
print(f"      R²:   {r2_simple:.4f}")

interpretacion = (
    f"\n   📝 Interpretación: Por cada hectárea adicional, se estiman "
    f"{modelo_lineal_simple.coef_[0]:.1f} toneladas más.\n"
    f"   Este modelo explica el {r2_simple * 100:.1f}% de la variabilidad. "
    f"Es un baseline útil pero NO captura los efectos no lineales del riego, "
    f"fertilizante ni antigüedad."
)
print(interpretacion)

# %% Celda 3: Modelo 2 - Regresión Lineal Múltiple
print("\n" + "=" * 70)
print("📐 MODELO 2: Regresión Lineal Múltiple")
print("=" * 70)
print("   Variables: todas las 5 features")
print("   Propósito: capturar efectos lineales de todas las variables\n")

modelo_lineal_multiple = LinearRegression()
modelo_lineal_multiple.fit(X_train_scaled, y_train)

y_pred_multiple = modelo_lineal_multiple.predict(X_test_scaled)

mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)

print("   Coeficientes (escalados):")
for feat, coef in zip(FEATURES, modelo_lineal_multiple.coef_):
    print(f"      {feat:30s}: {coef:+.4f}")
print(f"   Intercepto: {modelo_lineal_multiple.intercept_:.4f}")

print(f"\n   📏 Métricas en Test:")
print(f"      MSE:  {mse_multiple:.2f}")
print(f"      RMSE: {rmse_multiple:.2f} toneladas")
print(f"      MAE:  {mae_multiple:.2f} toneladas")
print(f"      R²:   {r2_multiple:.4f}")

mejora_r2 = r2_multiple - r2_simple
print(f"\n   📈 Mejora respecto a baseline: R² +{mejora_r2:.4f} ({mejora_r2 * 100:.1f} puntos %)")

# %% Celda 4: Modelo 3 - Red Neuronal
print("\n" + "=" * 70)
print("🧠 MODELO 3: Red Neuronal (Keras / TensorFlow)")
print("=" * 70)

print("""
   ARQUITECTURA PARA GERENCIA:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━
   La red neuronal es como un "cerebro artificial" que aprende las
   relaciones complejas entre las condiciones del lote y la producción.

   • Capa de entrada: recibe las 5 variables del lote
   • Capa oculta 1: 64 neuronas con activación ReLU
     (detecta patrones como el óptimo de riego)
   • Dropout 20%: evita "memorizar" datos en vez de aprender
   • Capa oculta 2: 32 neuronas con activación ReLU
     (refina las predicciones combinando los patrones)
   • Capa de salida: 1 neurona (toneladas estimadas)

   Ventaja sobre los modelos lineales: puede capturar que el riego
   tiene un punto óptimo, que el fertilizante tiene rendimientos
   decrecientes, y que la edad del viñedo tiene una curva en U invertida.
""")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers

# Fijar semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Construir modelo
modelo_nn = keras.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu', name='capa_patrones'),
    layers.Dropout(0.2, name='regularizacion'),
    layers.Dense(32, activation='relu', name='capa_refinamiento'),
    layers.Dense(1, name='prediccion'),
], name='AgroExport_RedNeuronal')

modelo_nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae'],
)

print("   Resumen del modelo:")
modelo_nn.summary()

# Entrenar
print("\n   🏋️ Entrenando red neuronal...")
history = modelo_nn.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True, verbose=1,
        ),
    ],
)

print(f"   ✅ Entrenamiento completado en {len(history.history['loss'])} épocas")

# Predicciones
y_pred_nn = modelo_nn.predict(X_test_scaled, verbose=0).flatten()

mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)

print(f"\n   📏 Métricas en Test:")
print(f"      MSE:  {mse_nn:.2f}")
print(f"      RMSE: {rmse_nn:.2f} toneladas")
print(f"      MAE:  {mae_nn:.2f} toneladas")
print(f"      R²:   {r2_nn:.4f}")

mejora_r2_nn = r2_nn - r2_multiple
print(f"\n   📈 Mejora respecto a lineal múltiple: R² +{mejora_r2_nn:.4f} ({mejora_r2_nn * 100:.1f} puntos %)")

# %% Celda 5: Tabla comparativa de métricas
print("\n" + "=" * 70)
print("📊 TABLA COMPARATIVA DE MODELOS")
print("=" * 70)

resultados = pd.DataFrame({
    'Modelo': ['Reg. Lineal Simple', 'Reg. Lineal Múltiple', 'Red Neuronal'],
    'MSE': [mse_simple, mse_multiple, mse_nn],
    'RMSE': [rmse_simple, rmse_multiple, rmse_nn],
    'MAE': [mae_simple, mae_multiple, mae_nn],
    'R²': [r2_simple, r2_multiple, r2_nn],
})

print("\n" + resultados.to_string(index=False, float_format='{:.4f}'.format))

# Destacar el mejor
mejor_idx = resultados['R²'].idxmax()
mejor_modelo = resultados.loc[mejor_idx, 'Modelo']
mejor_r2 = resultados.loc[mejor_idx, 'R²']
print(f"\n   🏆 Mejor modelo: {mejor_modelo} (R² = {mejor_r2:.4f})")

# %% Celda 6: Gráficos comparativos
print("\n📊 Generando gráficos comparativos...")

# --- Gráfico 1: Real vs Predicho para los 3 modelos ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
modelos_datos = [
    ("Reg. Lineal Simple", y_pred_simple, r2_simple),
    ("Reg. Lineal Múltiple", y_pred_multiple, r2_multiple),
    ("Red Neuronal", y_pred_nn, r2_nn),
]

for ax, (nombre, y_pred, r2) in zip(axes, modelos_datos):
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue', edgecolor='white', linewidth=0.3)
    lims = [
        min(y_test.min(), y_pred.min()) * 0.9,
        max(y_test.max(), y_pred.max()) * 1.1,
    ]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal (y=x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Valor Real (toneladas)")
    ax.set_ylabel("Valor Predicho (toneladas)")
    ax.set_title(f"{nombre}\nR² = {r2:.4f}", fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')

plt.suptitle("Comparación: Real vs Predicho", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "nb03_real_vs_predicho_comparacion.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 2: Barras comparativas de métricas ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

nombres = ['Lineal Simple', 'Lineal Múltiple', 'Red Neuronal']
colores = ['#ff7f0e', '#2ca02c', '#1f77b4']

# RMSE
bars1 = ax1.bar(nombres, [rmse_simple, rmse_multiple, rmse_nn], color=colores, edgecolor='white')
ax1.set_ylabel("RMSE (toneladas)")
ax1.set_title("Comparación de RMSE\n(menor es mejor)", fontweight='bold')
for bar, val in zip(bars1, [rmse_simple, rmse_multiple, rmse_nn]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.2f}', ha='center', fontweight='bold')

# R²
bars2 = ax2.bar(nombres, [r2_simple, r2_multiple, r2_nn], color=colores, edgecolor='white')
ax2.set_ylabel("R²")
ax2.set_title("Comparación de R²\n(mayor es mejor)", fontweight='bold')
ax2.set_ylim(0, 1.1)
for bar, val in zip(bars2, [r2_simple, r2_multiple, r2_nn]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "nb03_comparacion_metricas.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 3: Curva de pérdida de entrenamiento ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Pérdida (Train)', linewidth=2)
ax.plot(history.history['val_loss'], label='Pérdida (Validation)', linewidth=2, linestyle='--')
ax.set_xlabel("Época")
ax.set_ylabel("Pérdida (MSE)")
ax.set_title("Curva de Entrenamiento - Red Neuronal", fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "nb03_curva_perdida.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfico 4: Histograma de residuos ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
residuos_datos = [
    ("Lineal Simple", y_test - y_pred_simple),
    ("Lineal Múltiple", y_test - y_pred_multiple),
    ("Red Neuronal", y_test - y_pred_nn),
]

for ax, (nombre, residuos) in zip(axes, residuos_datos):
    ax.hist(residuos, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Residuo (toneladas)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Residuos: {nombre}\nμ={residuos.mean():.2f}, σ={residuos.std():.2f}",
                 fontweight='bold', fontsize=10)

plt.suptitle("Distribución de Residuos por Modelo", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "nb03_residuos.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Gráficos guardados en graficos/")

# %% Celda 7: Conclusiones y recomendación
print("\n" + "=" * 70)
print("📋 CONCLUSIONES Y RECOMENDACIÓN")
print("=" * 70)

print(f"""
   RESUMEN EJECUTIVO PARA GERENCIA
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Se evaluaron 3 modelos para predecir la producción de uva de mesa
   en la región Ica:

   1. Regresión Lineal Simple (baseline):
      • Solo usa hectáreas → R² = {r2_simple:.4f}
      • Útil como referencia mínima
      • No captura efectos de riego, fertilizante ni antigüedad

   2. Regresión Lineal Múltiple:
      • Usa las 5 variables → R² = {r2_multiple:.4f}
      • Mejora significativa sobre el baseline
      • Limitado: asume relaciones lineales

   3. Red Neuronal (RECOMENDADO ✅):
      • Usa las 5 variables con capas no lineales → R² = {r2_nn:.4f}
      • Captura el óptimo de riego, rendimientos decrecientes del
        fertilizante y la curva de edad del viñedo
      • Error promedio: {mae_nn:.1f} toneladas

   RECOMENDACIÓN:
   ━━━━━━━━━━━━━━
   La comparación mide el ajuste sobre datos semi-sintéticos, no la precisión
   esperable en fundos reales. La radiación sí procede de NASA POWER; MIDAGRI
   aporta estadísticas agregadas, y FAOSTAT no se descargó. Antes de usarlo
   en campo, hay que sustituir las cosechas e insumos simulados por registros reales.
""")

# Guardar modelos
print("💾 Los modelos se guardan desde src/modelos.py")
print("   Ejecutar: python src/modelos.py")

print("\n✅ Notebook 03 - Modelos completado")
