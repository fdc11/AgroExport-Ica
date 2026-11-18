"""
modelos.py - Módulo de Modelado Predictivo para AgroExport Ica
==============================================================
Implementa 3 modelos para predecir toneladas de cosecha de uva en Ica:
  1. Regresión Lineal Simple (baseline) - solo hectáreas
  2. Regresión Lineal Múltiple (intermedio) - todas las variables
  3. Red Neuronal Keras (avanzado) - arquitectura profunda con regularización

Estructura del CSV de entrada (data/processed/historial_cosechas_ica.csv):
  lote_id, hectareas, litros_riego, radiacion_solar_acumulada,
  kg_fertilizante, antiguedad_terreno, toneladas
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Suprimir warnings de TensorFlow para mantener la salida limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# Rutas del proyecto
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'historial_cosechas_ica.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Asegurar que el directorio de modelos exista
os.makedirs(MODELS_DIR, exist_ok=True)

# Variables predictoras
FEATURES = ['hectareas', 'litros_riego', 'radiacion_solar_acumulada',
            'kg_fertilizante', 'antiguedad_terreno']
TARGET = 'toneladas'


def cargar_datos(ruta=None):
    """Carga y retorna el dataset de cosechas de Ica.

    Parameters
    ----------
    ruta : str, optional
        Ruta al archivo CSV. Si no se proporciona, usa la ruta por defecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos de cosechas.
    """
    ruta = ruta or DATA_PATH
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {ruta}\n"
            "Ejecute primero src/generar_dataset.py para crear el dataset."
        )
    df = pd.read_csv(ruta)
    print(f"✓ Datos cargados: {df.shape[0]} lotes, {df.shape[1]} columnas")
    print(f"  Variables predictoras: {FEATURES}")
    print(f"  Variable objetivo: {TARGET}")
    return df


def preparar_datos(df, test_size=0.2, random_state=42):
    """Divide los datos en train/test y aplica escalado.

    Pipeline:
    - Split 80/20 con random_state=42
    - StandardScaler: fit SOLO en train (evitar data leakage)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con todas las columnas necesarias.
    test_size : float
        Proporción del conjunto de prueba.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict
        Diccionario con X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler, y los datos de
        variable simple.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    # Variable simple para regresión lineal baseline
    X_simple = df[['hectareas']].values

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_simple, X_test_simple = train_test_split(
        X_simple, test_size=test_size, random_state=random_state
    )

    # Escalado - fit SOLO en train para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✓ Datos preparados:")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")
    print(f"  Scaler ajustado SOLO con datos de entrenamiento (sin data leakage)")

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'X_train_simple': X_train_simple, 'X_test_simple': X_test_simple,
        'scaler': scaler
    }


def entrenar_regresion_simple(datos):
    """Modelo 1: Regresión Lineal Simple (baseline).

    Usa solo 'hectareas' para predecir 'toneladas'.
    Sirve como línea base para comparar modelos más complejos.

    Returns
    -------
    dict
        Modelo, predicciones y métricas.
    """
    print("\n" + "=" * 60)
    print("MODELO 1: Regresión Lineal Simple (Baseline)")
    print("  Predictor: hectareas → toneladas")
    print("=" * 60)

    modelo = LinearRegression()
    modelo.fit(datos['X_train_simple'], datos['y_train'])

    y_pred_train = modelo.predict(datos['X_train_simple'])
    y_pred_test = modelo.predict(datos['X_test_simple'])

    mse = mean_squared_error(datos['y_test'], y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(datos['y_test'], y_pred_test)
    mae = mean_absolute_error(datos['y_test'], y_pred_test)

    print(f"  Coeficiente (hectáreas): {modelo.coef_[0]:.4f}")
    print(f"  Intercepto:              {modelo.intercept_:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

    return {
        'modelo': modelo,
        'nombre': 'Regresión Lineal Simple',
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'metricas': {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}
    }


def entrenar_regresion_multiple(datos):
    """Modelo 2: Regresión Lineal Múltiple (intermedio).

    Usa todas las variables predictoras.

    Returns
    -------
    dict
        Modelo, predicciones y métricas.
    """
    print("\n" + "=" * 60)
    print("MODELO 2: Regresión Lineal Múltiple")
    print(f"  Predictores: {', '.join(FEATURES)} → toneladas")
    print("=" * 60)

    modelo = LinearRegression()
    modelo.fit(datos['X_train'], datos['y_train'])

    y_pred_train = modelo.predict(datos['X_train'])
    y_pred_test = modelo.predict(datos['X_test'])

    mse = mean_squared_error(datos['y_test'], y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(datos['y_test'], y_pred_test)
    mae = mean_absolute_error(datos['y_test'], y_pred_test)

    print(f"  Coeficientes:")
    for feat, coef in zip(FEATURES, modelo.coef_):
        print(f"    {feat:30s}: {coef:.6f}")
    print(f"  Intercepto: {modelo.intercept_:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'modelo_lineal_multiple.joblib')
    joblib.dump(modelo, ruta_modelo)
    print(f"  ✓ Modelo guardado en: {ruta_modelo}")

    return {
        'modelo': modelo,
        'nombre': 'Regresión Lineal Múltiple',
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'metricas': {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}
    }


def entrenar_red_neuronal(datos):
    """Modelo 3: Red Neuronal con Keras.

    Arquitectura:
        Input(5) → Dense(64, ReLU) → Dropout(0.2)
                  → Dense(32, ReLU) → Dropout(0.1)
                  → Dense(16, ReLU)
                  → Dense(1, linear)

    Configuración:
        - Optimizador: Adam(lr=0.001)
        - Loss: MSE
        - EarlyStopping(patience=20, restore_best_weights=True)
        - Epochs: 300 (early stopping cortará antes)
        - batch_size: 32

    Returns
    -------
    dict
        Modelo, predicciones, métricas e historial de entrenamiento.
    """
    print("\n" + "=" * 60)
    print("MODELO 3: Red Neuronal (Keras)")
    print("  Arquitectura: Input(5)→64→Drop(0.2)→32→Drop(0.1)→16→1")
    print("=" * 60)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Fijar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)

    # Construir arquitectura
    modelo = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ], name='red_neuronal_agroexport')

    # Compilar
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    modelo.summary()

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    # Entrenar con datos escalados
    print("\nEntrenando red neuronal...")
    historial = modelo.fit(
        datos['X_train_scaled'], datos['y_train'],
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Predicciones
    y_pred_train = modelo.predict(datos['X_train_scaled'], verbose=0).flatten()
    y_pred_test = modelo.predict(datos['X_test_scaled'], verbose=0).flatten()

    mse = mean_squared_error(datos['y_test'], y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(datos['y_test'], y_pred_test)
    mae = mean_absolute_error(datos['y_test'], y_pred_test)

    print(f"\n  Epochs completados: {len(historial.history['loss'])}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # Guardar modelo keras
    ruta_modelo = os.path.join(MODELS_DIR, 'modelo_red_neuronal.keras')
    modelo.save(ruta_modelo)
    print(f"  ✓ Modelo guardado en: {ruta_modelo}")

    return {
        'modelo': modelo,
        'nombre': 'Red Neuronal (Keras)',
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'metricas': {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae},
        'historial': historial.history
    }


def entrenar_todos_los_modelos(ruta_datos=None):
    """Función principal: entrena los 3 modelos y retorna resultados.

    Pipeline completo:
    1. Cargar datos
    2. Preparar (split + escalado)
    3. Entrenar Regresión Lineal Simple
    4. Entrenar Regresión Lineal Múltiple
    5. Entrenar Red Neuronal
    6. Guardar modelos y scaler
    7. Imprimir tabla comparativa

    Parameters
    ----------
    ruta_datos : str, optional
        Ruta al CSV de datos.

    Returns
    -------
    dict
        Diccionario con:
        - 'modelos': dict con los 3 modelos entrenados
        - 'predicciones': dict con predicciones train/test de cada modelo
        - 'metricas': dict con métricas de cada modelo
        - 'datos': dict con X_train, X_test, y_train, y_test, scaler
        - 'historial_nn': historial de entrenamiento de la red neuronal
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   AgroExport Ica - Entrenamiento de Modelos Predictivos ║")
    print("║   Predicción de toneladas de cosecha de uva             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Cargar datos
    df = cargar_datos(ruta_datos)

    # 2. Preparar datos
    datos = preparar_datos(df)

    # 3-5. Entrenar modelos
    resultado_simple = entrenar_regresion_simple(datos)
    resultado_multiple = entrenar_regresion_multiple(datos)
    resultado_nn = entrenar_red_neuronal(datos)

    # 6. Guardar scaler
    ruta_scaler = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(datos['scaler'], ruta_scaler)
    print(f"\n✓ Scaler guardado en: {ruta_scaler}")

    # 7. Tabla comparativa
    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA DE MODELOS")
    print("=" * 70)
    print(f"{'Modelo':<30s} {'MSE':>10s} {'RMSE':>10s} {'R²':>10s} {'MAE':>10s}")
    print("-" * 70)
    for res in [resultado_simple, resultado_multiple, resultado_nn]:
        m = res['metricas']
        print(f"{res['nombre']:<30s} {m['MSE']:>10.4f} {m['RMSE']:>10.4f} "
              f"{m['R2']:>10.4f} {m['MAE']:>10.4f}")
    print("=" * 70)

    # Construir resultado completo
    resultado = {
        'modelos': {
            'lineal_simple': resultado_simple['modelo'],
            'lineal_multiple': resultado_multiple['modelo'],
            'red_neuronal': resultado_nn['modelo']
        },
        'predicciones': {
            'lineal_simple': {
                'train': resultado_simple['y_pred_train'],
                'test': resultado_simple['y_pred_test']
            },
            'lineal_multiple': {
                'train': resultado_multiple['y_pred_train'],
                'test': resultado_multiple['y_pred_test']
            },
            'red_neuronal': {
                'train': resultado_nn['y_pred_train'],
                'test': resultado_nn['y_pred_test']
            }
        },
        'metricas': {
            'lineal_simple': resultado_simple['metricas'],
            'lineal_multiple': resultado_multiple['metricas'],
            'red_neuronal': resultado_nn['metricas']
        },
        'datos': datos,
        'historial_nn': resultado_nn.get('historial'),
        'y_test': datos['y_test'],
        'y_train': datos['y_train'],
        'df': df
    }

    print("\n✓ Entrenamiento completo. Todos los modelos guardados en models/")
    return resultado


def cargar_y_predecir(hectareas, litros_riego, radiacion, kg_fert, antiguedad):
    """Carga los modelos guardados y genera predicciones para nuevos datos.

    Carga el scaler, el modelo lineal múltiple y la red neuronal desde
    el directorio models/ y retorna las predicciones de ambos modelos.

    Parameters
    ----------
    hectareas : float
        Superficie del lote en hectáreas.
    litros_riego : float
        Litros de riego totales del lote.
    radiacion : float
        Radiación solar acumulada (kWh/m²/campaña).
    kg_fert : float
        Kilogramos de fertilizante aplicados.
    antiguedad : float
        Antigüedad del terreno en años.

    Returns
    -------
    dict
        Diccionario con predicciones de cada modelo y los datos de entrada.

    Example
    -------
    >>> resultado = cargar_y_predecir(
    ...     hectareas=15.0,
    ...     litros_riego=120000000.0,
    ...     radiacion=1100.0,
    ...     kg_fert=4500.0,
    ...     antiguedad=10.0
    ... )
    >>> print(f"Red Neuronal predice: {resultado['red_neuronal']:.2f} toneladas")
    """
    import tensorflow as tf

    # Rutas de archivos guardados
    ruta_scaler = os.path.join(MODELS_DIR, 'scaler.joblib')
    ruta_lineal = os.path.join(MODELS_DIR, 'modelo_lineal_multiple.joblib')
    ruta_nn = os.path.join(MODELS_DIR, 'modelo_red_neuronal.keras')

    # Verificar que los modelos existen
    for ruta, nombre in [(ruta_scaler, 'Scaler'), (ruta_lineal, 'Modelo Lineal'),
                         (ruta_nn, 'Red Neuronal')]:
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"{nombre} no encontrado en: {ruta}\n"
                "Ejecute primero entrenar_todos_los_modelos() para generar los modelos."
            )

    # Cargar artefactos
    scaler = joblib.load(ruta_scaler)
    modelo_lineal = joblib.load(ruta_lineal)
    modelo_nn = tf.keras.models.load_model(ruta_nn)

    # Preparar datos de entrada
    X_nuevo = np.array([[hectareas, litros_riego, radiacion, kg_fert, antiguedad]])

    # Predicción con regresión lineal simple (solo hectáreas)
    pred_simple = modelo_lineal.coef_[0] * hectareas + modelo_lineal.intercept_
    # Nota: usamos la ecuación directamente ya que el modelo simple no se guarda

    # Predicción con regresión lineal múltiple
    pred_multiple = modelo_lineal.predict(X_nuevo)[0]

    # Predicción con red neuronal (necesita escalado)
    X_scaled = scaler.transform(X_nuevo)
    pred_nn = modelo_nn.predict(X_scaled, verbose=0).flatten()[0]

    # Resultados
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║      AgroExport Ica - Predicción de Cosecha             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Datos de entrada:")
    print(f"    Hectáreas:              {hectareas:.1f} ha")
    print(f"    Litros de riego:        {litros_riego:,.0f} L")
    print(f"    Radiación solar acum.:  {radiacion:.1f} kWh/m²")
    print(f"    Fertilizante:           {kg_fert:.1f} kg")
    print(f"    Antigüedad terreno:     {antiguedad:.0f} años")
    print(f"\n  Predicciones:")
    print(f"    Regresión Lineal Múltiple: {pred_multiple:>10.2f} toneladas")
    print(f"    Red Neuronal:              {pred_nn:>10.2f} toneladas")
    print(f"\n  Rendimiento estimado (Red Neuronal): {pred_nn / hectareas:.2f} t/ha")

    return {
        'lineal_multiple': float(pred_multiple),
        'red_neuronal': float(pred_nn),
        'rendimiento_por_ha': float(pred_nn / hectareas),
        'entrada': {
            'hectareas': hectareas,
            'litros_riego': litros_riego,
            'radiacion': radiacion,
            'kg_fert': kg_fert,
            'antiguedad': antiguedad
        }
    }


# ==============================================================================
# Ejecución directa
# ==============================================================================
if __name__ == '__main__':
    resultado = entrenar_todos_los_modelos()

    # Ejemplo de predicción con un lote típico de Ica
    print("\n" + "=" * 60)
    print("EJEMPLO DE PREDICCIÓN")
    print("=" * 60)
    cargar_y_predecir(
        hectareas=20.0,
        litros_riego=160000000.0,
        radiacion=1120.0,
        kg_fert=6000.0,
        antiguedad=12.0
    )
