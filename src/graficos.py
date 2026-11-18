"""
graficos.py - Módulo de Visualización para AgroExport Ica
=========================================================
Genera 9 gráficos profesionales para análisis de modelos predictivos
de cosecha de uva en Ica, Perú.

Gráficos generados en graficos/:
  01 - Real vs Predicho (Lineal Simple)
  02 - Real vs Predicho (Lineal Múltiple)
  03 - Real vs Predicho (Red Neuronal)
  04 - Curva de Pérdida de Entrenamiento
  05 - Residuos de los 3 Modelos
  06 - Heatmap de Correlación
  07 - Comparación de Métricas
  08 - Importancia de Variables
  09 - Dispersión Hectáreas vs Toneladas
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar archivos
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Intentar importar seaborn
try:
    import seaborn as sns
    TIENE_SEABORN = True
except ImportError:
    TIENE_SEABORN = False

# ==============================================================================
# Configuración global de gráficos
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAFICOS_DIR = os.path.join(BASE_DIR, 'graficos')
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Paleta de colores profesional
COLORES = {
    'primario': '#2E86AB',      # Azul profesional
    'secundario': '#A23B72',    # Magenta
    'terciario': '#F18F01',     # Naranja
    'exito': '#2D936C',         # Verde
    'alerta': '#E74C3C',        # Rojo
    'fondo': '#F8F9FA',         # Gris claro
    'texto': '#2C3E50',         # Gris oscuro
    'linea_ideal': '#E74C3C',   # Rojo para línea ideal
    'train': '#2E86AB',         # Azul para training
    'val': '#F18F01',           # Naranja para validation
}

COLORES_MODELOS = ['#2E86AB', '#A23B72', '#F18F01']
NOMBRES_MODELOS = ['Regresión Lineal Simple', 'Regresión Lineal Múltiple', 'Red Neuronal']

# Configuración de matplotlib
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

DPI = 150


def _guardar(fig, nombre_archivo):
    """Guarda el gráfico y cierra la figura."""
    ruta = os.path.join(GRAFICOS_DIR, nombre_archivo)
    fig.savefig(ruta, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Guardado: {nombre_archivo}")
    return ruta


# ==============================================================================
# Gráfico 1-3: Real vs Predicho
# ==============================================================================

def _grafico_real_vs_predicho(y_real, y_pred, titulo, nombre_archivo, color):
    """Gráfico scatter de valores reales vs predichos con línea ideal y=x."""
    fig, ax = plt.subplots(figsize=(10, 8))

    r2 = r2_score(y_real, y_pred)

    # Scatter
    ax.scatter(y_real, y_pred, alpha=0.5, color=color, edgecolors='white',
               linewidth=0.5, s=50, label='Predicciones')

    # Línea ideal y=x
    lim_min = min(y_real.min(), y_pred.min()) * 0.9
    lim_max = max(y_real.max(), y_pred.max()) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--',
            color=COLORES['linea_ideal'], linewidth=2, label='Línea ideal (y=x)',
            zorder=5)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Valores Reales (toneladas)', fontweight='bold')
    ax.set_ylabel('Valores Predichos (toneladas)', fontweight='bold')
    ax.set_title(f'{titulo}\nR² = {r2:.4f}', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')

    fig.tight_layout()
    return _guardar(fig, nombre_archivo)


def grafico_01_real_vs_predicho_lineal_simple(resultado):
    """01 - Real vs Predicho: Regresión Lineal Simple."""
    return _grafico_real_vs_predicho(
        resultado['y_test'],
        resultado['predicciones']['lineal_simple']['test'],
        'Real vs Predicho - Regresión Lineal Simple',
        '01_real_vs_predicho_lineal_simple.png',
        COLORES_MODELOS[0]
    )


def grafico_02_real_vs_predicho_lineal_multiple(resultado):
    """02 - Real vs Predicho: Regresión Lineal Múltiple."""
    return _grafico_real_vs_predicho(
        resultado['y_test'],
        resultado['predicciones']['lineal_multiple']['test'],
        'Real vs Predicho - Regresión Lineal Múltiple',
        '02_real_vs_predicho_lineal_multiple.png',
        COLORES_MODELOS[1]
    )


def grafico_03_real_vs_predicho_red_neuronal(resultado):
    """03 - Real vs Predicho: Red Neuronal."""
    return _grafico_real_vs_predicho(
        resultado['y_test'],
        resultado['predicciones']['red_neuronal']['test'],
        'Real vs Predicho - Red Neuronal (Keras)',
        '03_real_vs_predicho_red_neuronal.png',
        COLORES_MODELOS[2]
    )


# ==============================================================================
# Gráfico 4: Curva de pérdida de entrenamiento
# ==============================================================================

def grafico_04_curva_perdida(resultado):
    """04 - Curva de pérdida de entrenamiento de la red neuronal."""
    historial = resultado.get('historial_nn')
    if historial is None:
        print("  ⚠ No se encontró historial de entrenamiento. Omitiendo gráfico 04.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = range(1, len(historial['loss']) + 1)

    ax.plot(epochs, historial['loss'], color=COLORES['train'],
            linewidth=2, label='Pérdida - Entrenamiento', alpha=0.8)
    ax.plot(epochs, historial['val_loss'], color=COLORES['val'],
            linewidth=2, label='Pérdida - Validación', alpha=0.8)

    # Marcar el mejor epoch
    mejor_epoch = np.argmin(historial['val_loss']) + 1
    mejor_val_loss = min(historial['val_loss'])
    ax.axvline(x=mejor_epoch, color=COLORES['exito'], linestyle=':', alpha=0.7,
               label=f'Mejor epoch: {mejor_epoch}')
    ax.scatter([mejor_epoch], [mejor_val_loss], color=COLORES['exito'],
               s=100, zorder=5, marker='*')

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Pérdida (MSE)', fontweight='bold')
    ax.set_title('Curva de Pérdida durante el Entrenamiento\nRed Neuronal - AgroExport Ica',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)

    # Escala logarítmica si hay mucha variación
    if max(historial['loss']) / min(historial['loss']) > 50:
        ax.set_yscale('log')
        ax.set_ylabel('Pérdida (MSE, escala log)', fontweight='bold')

    fig.tight_layout()
    return _guardar(fig, '04_curva_perdida_entrenamiento.png')


# ==============================================================================
# Gráfico 5: Histograma de residuos de los 3 modelos
# ==============================================================================

def grafico_05_residuos(resultado):
    """05 - Subplot con histograma de residuos de los 3 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    claves = ['lineal_simple', 'lineal_multiple', 'red_neuronal']

    for ax, clave, nombre, color in zip(axes, claves, NOMBRES_MODELOS, COLORES_MODELOS):
        y_pred = resultado['predicciones'][clave]['test']
        residuos = resultado['y_test'] - y_pred

        ax.hist(residuos, bins=25, color=color, alpha=0.7, edgecolor='white',
                linewidth=0.5, density=True)

        # Línea de media
        media = np.mean(residuos)
        std = np.std(residuos)
        ax.axvline(media, color=COLORES['alerta'], linestyle='--', linewidth=2,
                   label=f'Media: {media:.2f}')
        ax.axvline(media + std, color=COLORES['texto'], linestyle=':', alpha=0.5)
        ax.axvline(media - std, color=COLORES['texto'], linestyle=':', alpha=0.5)

        ax.set_xlabel('Residuo (toneladas)', fontweight='bold')
        ax.set_ylabel('Densidad', fontweight='bold')
        ax.set_title(f'Residuos\n{nombre}', fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle('Distribución de Residuos - Comparación de Modelos',
                 fontweight='bold', fontsize=16, y=1.02)
    fig.tight_layout()
    return _guardar(fig, '05_residuos_tres_modelos.png')


# ==============================================================================
# Gráfico 6: Heatmap de correlación
# ==============================================================================

def grafico_06_heatmap_correlacion(resultado):
    """06 - Heatmap de correlación de todas las variables."""
    df = resultado['df']
    columnas_numericas = ['hectareas', 'litros_riego', 'radiacion_solar_acumulada',
                          'kg_fertilizante', 'antiguedad_terreno', 'toneladas']
    corr = df[columnas_numericas].corr()

    # Etiquetas en español
    etiquetas = ['Hectáreas', 'Litros Riego', 'Radiación Solar',
                 'Kg Fertilizante', 'Antigüedad', 'Toneladas']

    fig, ax = plt.subplots(figsize=(10, 8))

    if TIENE_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1,
                    xticklabels=etiquetas, yticklabels=etiquetas,
                    square=True, linewidths=0.5,
                    cbar_kws={'label': 'Coeficiente de Correlación'},
                    ax=ax)
    else:
        # Fallback con matplotlib imshow
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(etiquetas)))
        ax.set_yticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=45, ha='right')
        ax.set_yticklabels(etiquetas)

        # Añadir valores numéricos
        for i in range(len(etiquetas)):
            for j in range(len(etiquetas)):
                valor = corr.values[i, j]
                color_texto = 'white' if abs(valor) > 0.5 else 'black'
                ax.text(j, i, f'{valor:.3f}', ha='center', va='center',
                        color=color_texto, fontsize=10)

        fig.colorbar(im, ax=ax, label='Coeficiente de Correlación', shrink=0.8)

    ax.set_title('Mapa de Correlación - Variables de Producción Agrícola\nIca, Perú',
                 fontweight='bold', fontsize=14, pad=20)

    fig.tight_layout()
    return _guardar(fig, '06_heatmap_correlacion.png')


# ==============================================================================
# Gráfico 7: Comparación de métricas (barras agrupadas)
# ==============================================================================

def grafico_07_comparacion_metricas(resultado):
    """07 - Barras agrupadas comparando RMSE y R² de los 3 modelos."""
    metricas = resultado['metricas']
    claves = ['lineal_simple', 'lineal_multiple', 'red_neuronal']

    rmse_vals = [metricas[c]['RMSE'] for c in claves]
    r2_vals = [metricas[c]['R2'] for c in claves]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(NOMBRES_MODELOS))
    ancho = 0.6

    # RMSE
    barras_rmse = ax1.bar(x, rmse_vals, ancho, color=COLORES_MODELOS, alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('RMSE (toneladas)', fontweight='bold')
    ax1.set_title('Comparación de RMSE\n(Menor es mejor)', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(NOMBRES_MODELOS, rotation=15, ha='right')

    # Etiquetas de valor
    for barra, val in zip(barras_rmse, rmse_vals):
        ax1.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Marcar el mejor (menor RMSE)
    mejor_idx = np.argmin(rmse_vals)
    barras_rmse[mejor_idx].set_edgecolor(COLORES['exito'])
    barras_rmse[mejor_idx].set_linewidth(3)

    # R²
    barras_r2 = ax2.bar(x, r2_vals, ancho, color=COLORES_MODELOS, alpha=0.85,
                         edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('R² (Coeficiente de Determinación)', fontweight='bold')
    ax2.set_title('Comparación de R²\n(Mayor es mejor)', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(NOMBRES_MODELOS, rotation=15, ha='right')
    ax2.set_ylim(0, 1.1)

    # Etiquetas de valor
    for barra, val in zip(barras_r2, r2_vals):
        ax2.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.02,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Marcar el mejor (mayor R²)
    mejor_idx_r2 = np.argmax(r2_vals)
    barras_r2[mejor_idx_r2].set_edgecolor(COLORES['exito'])
    barras_r2[mejor_idx_r2].set_linewidth(3)

    fig.suptitle('Comparación de Métricas entre Modelos - AgroExport Ica',
                 fontweight='bold', fontsize=16, y=1.02)
    fig.tight_layout()
    return _guardar(fig, '07_comparacion_metricas.png')


# ==============================================================================
# Gráfico 8: Importancia de variables
# ==============================================================================

def grafico_08_importancia_variables(importancia):
    """08 - Barras horizontales de importancia por permutación.

    Parameters
    ----------
    importancia : dict
        Diccionario de importancia retornado por importancia_variables_permutacion().
    """
    # Etiquetas en español
    traducciones = {
        'hectareas': 'Hectáreas',
        'litros_riego': 'Litros de Riego',
        'radiacion_solar_acumulada': 'Radiación Solar Acumulada',
        'kg_fertilizante': 'Kg Fertilizante',
        'antiguedad_terreno': 'Antigüedad del Terreno'
    }

    # Ordenar de menor a mayor para que el más importante quede arriba
    nombres = list(importancia.keys())
    valores = [importancia[n]['media'] for n in nombres]
    errores = [importancia[n]['std'] for n in nombres]

    # Ordenar ascendente (el mayor quedará arriba en barh)
    indices_ordenados = np.argsort(valores)
    nombres = [nombres[i] for i in indices_ordenados]
    valores = [valores[i] for i in indices_ordenados]
    errores = [errores[i] for i in indices_ordenados]

    etiquetas = [traducciones.get(n, n) for n in nombres]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Colores en gradiente
    n_vars = len(nombres)
    colores = plt.cm.viridis(np.linspace(0.3, 0.9, n_vars))

    barras = ax.barh(range(n_vars), valores, xerr=errores,
                     color=colores, alpha=0.85, edgecolor='white',
                     linewidth=1, capsize=5, error_kw={'linewidth': 1.5})

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(etiquetas, fontsize=12)
    ax.set_xlabel('Incremento en MSE (Importancia por Permutación)', fontweight='bold')
    ax.set_title('Importancia de Variables - Red Neuronal\nMétodo: Permutación sobre conjunto de test',
                 fontweight='bold', fontsize=14)

    # Valores en las barras
    for barra, val in zip(barras, valores):
        ancho = barra.get_width()
        if ancho > 0:
            ax.text(ancho + max(errores) * 0.3, barra.get_y() + barra.get_height() / 2,
                    f'{val:.2f}', va='center', fontweight='bold', fontsize=10)

    fig.tight_layout()
    return _guardar(fig, '08_importancia_variables.png')


# ==============================================================================
# Gráfico 9: Dispersión hectáreas vs toneladas
# ==============================================================================

def grafico_09_dispersion_hectareas_toneladas(resultado):
    """09 - Scatter de hectáreas vs toneladas con línea de regresión."""
    df = resultado['df']

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(df['hectareas'], df['toneladas'], alpha=0.5,
               color=COLORES['primario'], edgecolors='white',
               linewidth=0.5, s=50, label='Lotes')

    # Línea de regresión
    X_ha = df['hectareas'].values.reshape(-1, 1)
    y_ton = df['toneladas'].values
    reg = LinearRegression().fit(X_ha, y_ton)
    x_linea = np.linspace(df['hectareas'].min(), df['hectareas'].max(), 100)
    y_linea = reg.predict(x_linea.reshape(-1, 1))

    r2 = r2_score(y_ton, reg.predict(X_ha))

    ax.plot(x_linea, y_linea, color=COLORES['alerta'], linewidth=2.5,
            label=f'Regresión lineal (R²={r2:.4f})', zorder=5)

    # Ecuación en el gráfico
    ecuacion = f'y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}'
    ax.text(0.05, 0.95, ecuacion, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Hectáreas', fontweight='bold', fontsize=13)
    ax.set_ylabel('Toneladas de Cosecha', fontweight='bold', fontsize=13)
    ax.set_title('Relación entre Superficie y Producción\nViñedos de Ica, Perú',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)

    fig.tight_layout()
    return _guardar(fig, '09_dispersion_hectareas_vs_toneladas.png')


# ==============================================================================
# Función principal: generar todos los gráficos
# ==============================================================================

def generar_todos_los_graficos(resultado, importancia=None):
    """Genera los 9 gráficos del proyecto y los guarda en graficos/.

    Parameters
    ----------
    resultado : dict
        Diccionario retornado por entrenar_todos_los_modelos() de modelos.py.
    importancia : dict, optional
        Diccionario de importancia retornado por importancia_variables_permutacion().
        Si no se proporciona, se calcula automáticamente.

    Returns
    -------
    list
        Lista de rutas de los gráficos generados.
    """
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║      Generación de Gráficos - AgroExport Ica           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Directorio de salida: {GRAFICOS_DIR}")
    print(f"  Resolución: {DPI} DPI")
    print(f"  Seaborn disponible: {'Sí' if TIENE_SEABORN else 'No (usando matplotlib)'}")
    print()

    rutas = []

    # Gráficos 1-3: Real vs Predicho
    rutas.append(grafico_01_real_vs_predicho_lineal_simple(resultado))
    rutas.append(grafico_02_real_vs_predicho_lineal_multiple(resultado))
    rutas.append(grafico_03_real_vs_predicho_red_neuronal(resultado))

    # Gráfico 4: Curva de pérdida
    ruta_04 = grafico_04_curva_perdida(resultado)
    if ruta_04:
        rutas.append(ruta_04)

    # Gráfico 5: Residuos
    rutas.append(grafico_05_residuos(resultado))

    # Gráfico 6: Heatmap correlación
    rutas.append(grafico_06_heatmap_correlacion(resultado))

    # Gráfico 7: Comparación métricas
    rutas.append(grafico_07_comparacion_metricas(resultado))

    # Gráfico 8: Importancia de variables
    if importancia is None:
        # Calcular importancia si no se proporcionó
        from evaluacion import importancia_variables_permutacion
        importancia = importancia_variables_permutacion(
            modelo_nn=resultado['modelos']['red_neuronal'],
            X_test_scaled=resultado['datos']['X_test_scaled'],
            y_test=resultado['y_test']
        )
    rutas.append(grafico_08_importancia_variables(importancia))

    # Gráfico 9: Dispersión hectáreas vs toneladas
    rutas.append(grafico_09_dispersion_hectareas_toneladas(resultado))

    print(f"\n✓ {len(rutas)} gráficos generados exitosamente en graficos/")
    return rutas


# ==============================================================================
# Ejecución directa
# ==============================================================================
if __name__ == '__main__':
    from modelos import entrenar_todos_los_modelos
    from evaluacion import importancia_variables_permutacion

    print("Entrenando modelos para generación de gráficos...")
    resultado = entrenar_todos_los_modelos()

    print("\nCalculando importancia de variables...")
    importancia = importancia_variables_permutacion(
        modelo_nn=resultado['modelos']['red_neuronal'],
        X_test_scaled=resultado['datos']['X_test_scaled'],
        y_test=resultado['y_test']
    )

    print("\nGenerando gráficos...")
    rutas = generar_todos_los_graficos(resultado, importancia)

    print("\n" + "=" * 60)
    print("GRÁFICOS GENERADOS:")
    print("=" * 60)
    for ruta in rutas:
        print(f"  📊 {os.path.basename(ruta)}")
