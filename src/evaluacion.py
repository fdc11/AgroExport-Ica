"""
evaluacion.py - Módulo de Evaluación de Modelos para AgroExport Ica
===================================================================
Evalúa y compara los 3 modelos predictivos:
  - Métricas: MSE, RMSE, R², MAE
  - Tabla comparativa formateada
  - Importancia de variables por permutación (red neuronal)
  - Análisis de residuos con test de normalidad

Requiere que los modelos hayan sido entrenados previamente con modelos.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ==============================================================================
# Funciones de evaluación de métricas
# ==============================================================================

def calcular_metricas(y_real, y_pred):
    """Calcula MSE, RMSE, R² y MAE para un conjunto de predicciones.

    Parameters
    ----------
    y_real : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.

    Returns
    -------
    dict
        Diccionario con MSE, RMSE, R2 y MAE.
    """
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}


def evaluar_todos_los_modelos(resultado):
    """Evalúa los 3 modelos en el conjunto de test y genera tabla comparativa.

    Parameters
    ----------
    resultado : dict
        Diccionario retornado por entrenar_todos_los_modelos() de modelos.py.
        Debe contener 'predicciones', 'y_test' y 'metricas'.

    Returns
    -------
    dict
        Diccionario con métricas de cada modelo y DataFrame comparativo.
    """
    y_test = resultado['y_test']

    nombres_modelos = {
        'lineal_simple': 'Regresión Lineal Simple',
        'lineal_multiple': 'Regresión Lineal Múltiple',
        'red_neuronal': 'Red Neuronal (Keras)'
    }

    metricas_todos = {}

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║          EVALUACIÓN COMPLETA DE MODELOS - AgroExport Ica       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    for clave, nombre in nombres_modelos.items():
        y_pred = resultado['predicciones'][clave]['test']
        metricas = calcular_metricas(y_test, y_pred)
        metricas_todos[clave] = metricas

    # Tabla comparativa formateada
    _imprimir_tabla_comparativa(metricas_todos, nombres_modelos)

    # DataFrame para uso programático
    df_metricas = pd.DataFrame({
        nombre: metricas_todos[clave]
        for clave, nombre in nombres_modelos.items()
    }).T
    df_metricas.index.name = 'Modelo'

    return {
        'metricas': metricas_todos,
        'nombres': nombres_modelos,
        'df_comparativo': df_metricas
    }


def _imprimir_tabla_comparativa(metricas_todos, nombres_modelos):
    """Imprime tabla comparativa formateada de métricas de los 3 modelos."""

    print("\n┌─────────────────────────────────┬────────────┬────────────┬────────────┬────────────┐")
    print(f"│ {'Modelo':<31s} │ {'MSE':^10s} │ {'RMSE':^10s} │ {'R²':^10s} │ {'MAE':^10s} │")
    print("├─────────────────────────────────┼────────────┼────────────┼────────────┼────────────┤")

    for clave, nombre in nombres_modelos.items():
        m = metricas_todos[clave]
        print(f"│ {nombre:<31s} │ {m['MSE']:>10.4f} │ {m['RMSE']:>10.4f} │ "
              f"{m['R2']:>10.4f} │ {m['MAE']:>10.4f} │")

    print("└─────────────────────────────────┴────────────┴────────────┴────────────┴────────────┘")

    # Mejor modelo
    mejor_clave = min(metricas_todos, key=lambda k: metricas_todos[k]['RMSE'])
    mejor_nombre = nombres_modelos[mejor_clave]
    mejor_r2 = metricas_todos[mejor_clave]['R2']
    mejor_rmse = metricas_todos[mejor_clave]['RMSE']

    print(f"\n  🏆 Mejor modelo por RMSE: {mejor_nombre}")
    print(f"     RMSE = {mejor_rmse:.4f} | R² = {mejor_r2:.4f}")

    # Mejora respecto al baseline
    rmse_base = metricas_todos['lineal_simple']['RMSE']
    if mejor_clave != 'lineal_simple':
        mejora = ((rmse_base - mejor_rmse) / rmse_base) * 100
        print(f"     Mejora sobre baseline: {mejora:.1f}% en RMSE")


# ==============================================================================
# Importancia de variables por permutación
# ==============================================================================

def importancia_variables_permutacion(modelo_nn, X_test_scaled, y_test,
                                       nombres_variables=None, n_repeticiones=10,
                                       random_state=42):
    """Calcula la importancia de variables por permutación para la red neuronal.

    Para cada variable, permuta sus valores en el conjunto de test y mide
    cuánto empeora el MSE. Mayor empeoramiento = mayor importancia.

    Parameters
    ----------
    modelo_nn : keras.Model
        Red neuronal entrenada.
    X_test_scaled : np.ndarray
        Datos de test escalados.
    y_test : np.ndarray
        Valores reales de test.
    nombres_variables : list, optional
        Nombres de las variables. Por defecto usa los del proyecto.
    n_repeticiones : int
        Número de permutaciones por variable.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict
        Diccionario con importancia media y std por variable,
        ordenado de mayor a menor importancia.
    """
    if nombres_variables is None:
        nombres_variables = ['hectareas', 'litros_riego', 'radiacion_solar_acumulada',
                             'kg_fertilizante', 'antiguedad_terreno']

    rng = np.random.RandomState(random_state)

    # MSE base (sin permutación)
    y_pred_base = modelo_nn.predict(X_test_scaled, verbose=0).flatten()
    mse_base = mean_squared_error(y_test, y_pred_base)

    importancias = {}

    print("\n" + "=" * 60)
    print("IMPORTANCIA DE VARIABLES POR PERMUTACIÓN (Red Neuronal)")
    print("=" * 60)
    print(f"  MSE base (sin permutación): {mse_base:.4f}")
    print(f"  Repeticiones por variable:  {n_repeticiones}")
    print()

    for i, nombre in enumerate(nombres_variables):
        incrementos_mse = []

        for _ in range(n_repeticiones):
            X_perm = X_test_scaled.copy()
            rng.shuffle(X_perm[:, i])  # Permutar columna i

            y_pred_perm = modelo_nn.predict(X_perm, verbose=0).flatten()
            mse_perm = mean_squared_error(y_test, y_pred_perm)
            incrementos_mse.append(mse_perm - mse_base)

        media = np.mean(incrementos_mse)
        std = np.std(incrementos_mse)
        importancias[nombre] = {'media': media, 'std': std}

    # Ordenar por importancia (mayor incremento MSE = más importante)
    importancias_ordenadas = dict(
        sorted(importancias.items(), key=lambda x: x[1]['media'], reverse=True)
    )

    # Imprimir resultados
    print(f"  {'Variable':<35s} {'Importancia (ΔMSE)':>18s} {'± Std':>10s}")
    print("  " + "-" * 65)
    for nombre, vals in importancias_ordenadas.items():
        barra = "█" * int(min(vals['media'] / max(v['media'] for v in importancias.values()) * 20, 20))
        print(f"  {nombre:<35s} {vals['media']:>18.4f} {vals['std']:>10.4f}  {barra}")

    return importancias_ordenadas


# ==============================================================================
# Análisis de residuos
# ==============================================================================

def analisis_residuos(resultado):
    """Realiza análisis de residuos para los 3 modelos.

    Calcula:
    - Media y desviación estándar de residuos
    - Test de normalidad de Shapiro-Wilk (si scipy disponible)
    - Rango de residuos (min, max)

    Parameters
    ----------
    resultado : dict
        Diccionario retornado por entrenar_todos_los_modelos().

    Returns
    -------
    dict
        Diccionario con análisis de residuos de cada modelo.
    """
    y_test = resultado['y_test']

    nombres_modelos = {
        'lineal_simple': 'Regresión Lineal Simple',
        'lineal_multiple': 'Regresión Lineal Múltiple',
        'red_neuronal': 'Red Neuronal (Keras)'
    }

    # Intentar importar scipy para test de normalidad
    tiene_scipy = False
    try:
        from scipy import stats
        tiene_scipy = True
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print("ANÁLISIS DE RESIDUOS")
    print("=" * 70)

    if not tiene_scipy:
        print("  ⚠ scipy no disponible. Test de Shapiro-Wilk omitido.")

    residuos_todos = {}

    for clave, nombre in nombres_modelos.items():
        y_pred = resultado['predicciones'][clave]['test']
        residuos = y_test - y_pred

        analisis = {
            'residuos': residuos,
            'media': np.mean(residuos),
            'std': np.std(residuos),
            'min': np.min(residuos),
            'max': np.max(residuos),
            'mediana': np.median(residuos),
            'asimetria': float(pd.Series(residuos).skew()),
            'curtosis': float(pd.Series(residuos).kurtosis())
        }

        # Test de Shapiro-Wilk
        if tiene_scipy:
            # Usar máximo 5000 observaciones (límite del test)
            muestra = residuos[:5000] if len(residuos) > 5000 else residuos
            stat, p_valor = stats.shapiro(muestra)
            analisis['shapiro_stat'] = stat
            analisis['shapiro_p'] = p_valor
            analisis['es_normal'] = p_valor > 0.05

        residuos_todos[clave] = analisis

        # Imprimir
        print(f"\n  {nombre}")
        print(f"  {'─' * 50}")
        print(f"    Media de residuos:    {analisis['media']:>12.4f}")
        print(f"    Std de residuos:      {analisis['std']:>12.4f}")
        print(f"    Mediana:              {analisis['mediana']:>12.4f}")
        print(f"    Rango:                [{analisis['min']:.4f}, {analisis['max']:.4f}]")
        print(f"    Asimetría (skew):     {analisis['asimetria']:>12.4f}")
        print(f"    Curtosis:             {analisis['curtosis']:>12.4f}")

        if tiene_scipy:
            estado = "✓ Normal" if analisis['es_normal'] else "✗ No normal"
            print(f"    Shapiro-Wilk stat:    {analisis['shapiro_stat']:>12.4f}")
            print(f"    Shapiro-Wilk p-valor: {analisis['shapiro_p']:>12.6f}  ({estado}, α=0.05)")

    return residuos_todos


def evaluar_completo(resultado):
    """Ejecuta evaluación completa: métricas, importancia y residuos.

    Función de conveniencia que ejecuta todas las evaluaciones en secuencia.

    Parameters
    ----------
    resultado : dict
        Diccionario retornado por entrenar_todos_los_modelos().

    Returns
    -------
    dict
        Diccionario con resultados de métricas, importancia y residuos.
    """
    # 1. Métricas comparativas
    eval_metricas = evaluar_todos_los_modelos(resultado)

    # 2. Importancia de variables (red neuronal)
    importancia = importancia_variables_permutacion(
        modelo_nn=resultado['modelos']['red_neuronal'],
        X_test_scaled=resultado['datos']['X_test_scaled'],
        y_test=resultado['y_test']
    )

    # 3. Análisis de residuos
    residuos = analisis_residuos(resultado)

    return {
        'metricas': eval_metricas,
        'importancia_variables': importancia,
        'residuos': residuos
    }


# ==============================================================================
# Ejecución directa
# ==============================================================================
if __name__ == '__main__':
    # Importar y ejecutar modelos primero
    from modelos import entrenar_todos_los_modelos

    print("Entrenando modelos para evaluación...")
    resultado = entrenar_todos_los_modelos()

    print("\n\n" + "═" * 70)
    print("    INICIANDO EVALUACIÓN COMPLETA")
    print("═" * 70)

    evaluacion = evaluar_completo(resultado)

    print("\n\n✓ Evaluación completa finalizada.")
