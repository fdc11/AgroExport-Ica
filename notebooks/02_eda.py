# %% [markdown]
# # 02 - Análisis Exploratorio de Datos (EDA)
#
# **Proyecto**: AgroExport Ica
# **Objetivo**: Explorar y visualizar el dataset de cosechas de uva de mesa
# en la región Ica para entender las relaciones entre variables antes del modelado.

# %% Celda 1: Cargar datos
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="viridis")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "historial_cosechas_ica.csv"
GRAFICOS_DIR = PROJECT_ROOT / "graficos"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Proyecto: {PROJECT_ROOT}")
print(f"📄 Datos: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"📋 Columnas: {list(df.columns)}")

# Agregar rendimiento por hectárea
if 'toneladas' in df.columns and 'hectareas' in df.columns:
    df['rendimiento_ha'] = df['toneladas'] / df['hectareas']

# %% Celda 2: df.describe() y df.info()
print("\n" + "=" * 70)
print("📊 INFORMACIÓN DEL DATASET")
print("=" * 70)

print("\n--- df.info() ---")
df.info()

print("\n\n--- df.describe() ---")
print(df.describe().round(2).to_string())

print("\n\n--- Valores nulos ---")
print(df.isnull().sum().to_string())

print("\n--- Tipos de datos ---")
print(df.dtypes.to_string())

# %% Celda 3: Distribuciones de cada variable (histogramas)
print("\n📊 Generando histogramas de distribución...")

# Variables numéricas a graficar (excluir lote_id)
vars_numericas = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'lote_id']

n_vars = len(vars_numericas)
ncols = 3
nrows = (n_vars + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

colores = plt.cm.viridis(np.linspace(0.2, 0.8, n_vars))

for i, var in enumerate(vars_numericas):
    ax = axes[i]
    ax.hist(df[var], bins=30, color=colores[i], edgecolor='white', alpha=0.85)
    ax.set_title(f"Distribución de {var}", fontsize=11, fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel("Frecuencia")

    # Línea de media
    media = df[var].mean()
    ax.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'Media: {media:.1f}')
    ax.legend(fontsize=8)

# Ocultar ejes sobrantes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribuciones de Variables - AgroExport Ica", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "eda_distribuciones.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Guardado: graficos/eda_distribuciones.png")

# %% Celda 4: Heatmap de correlación
print("\n📊 Generando heatmap de correlación...")

# Seleccionar solo variables numéricas relevantes
vars_corr = [c for c in vars_numericas if c not in ['lote_id']]
corr_matrix = df[vars_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))

if HAS_SEABORN:
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
else:
    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(vars_corr)))
    ax.set_yticks(range(len(vars_corr)))
    ax.set_xticklabels(vars_corr, rotation=45, ha='right')
    ax.set_yticklabels(vars_corr)
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Annotate
    for i_row in range(len(vars_corr)):
        for j_col in range(len(vars_corr)):
            ax.text(j_col, i_row, f"{corr_matrix.iloc[i_row, j_col]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr_matrix.iloc[i_row, j_col]) > 0.5 else "black")

ax.set_title("Matriz de Correlación - Variables de Cosecha Ica", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "eda_heatmap_correlacion.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Guardado: graficos/eda_heatmap_correlacion.png")

# Mostrar correlaciones con la variable objetivo
print("\n📈 Correlación con 'toneladas':")
if 'toneladas' in corr_matrix.columns:
    corr_target = corr_matrix['toneladas'].drop('toneladas').sort_values(ascending=False)
    for var, corr in corr_target.items():
        barra = "█" * int(abs(corr) * 20)
        signo = "+" if corr > 0 else "-"
        print(f"  {var:30s} {signo}{abs(corr):.3f}  {barra}")

# %% Celda 5: Scatter hectáreas vs toneladas
print("\n📊 Generando scatter hectáreas vs toneladas...")

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    df['hectareas'], df['toneladas'],
    c=df.get('rendimiento_ha', df['toneladas'] / df['hectareas']),
    cmap='RdYlGn', alpha=0.6, s=30, edgecolor='gray', linewidth=0.3,
)
plt.colorbar(scatter, ax=ax, label='Rendimiento (t/ha)')

# Línea de tendencia
z = np.polyfit(df['hectareas'], df['toneladas'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['hectareas'].min(), df['hectareas'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Tendencia lineal (pendiente={z[0]:.1f})')

# Objetivo de rendimiento del escenario semi-sintético, no promedio oficial.
ax.plot(x_line, x_line * 22, 'b:', linewidth=1.5, alpha=0.5, label='Objetivo sintético: 22 t/ha')

ax.set_xlabel("Hectáreas", fontsize=12)
ax.set_ylabel("Toneladas producidas", fontsize=12)
ax.set_title("Relación entre Superficie y Producción Total", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "eda_scatter_hectareas_toneladas.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Guardado: graficos/eda_scatter_hectareas_toneladas.png")

# %% Celda 6: Boxplots por variable
print("\n📊 Generando boxplots...")

vars_box = [c for c in vars_numericas if c not in ['lote_id', 'rendimiento_ha']]

fig, axes = plt.subplots(1, len(vars_box), figsize=(3.5 * len(vars_box), 6))
if len(vars_box) == 1:
    axes = [axes]

colores_box = plt.cm.Set2(np.linspace(0, 1, len(vars_box)))

for i, var in enumerate(vars_box):
    bp = axes[i].boxplot(
        df[var].dropna(), patch_artist=True, widths=0.6,
        boxprops=dict(facecolor=colores_box[i], alpha=0.7),
        medianprops=dict(color='darkred', linewidth=2),
        flierprops=dict(marker='o', markersize=4, alpha=0.4),
    )
    axes[i].set_title(var, fontsize=10, fontweight='bold')
    axes[i].set_ylabel(var)

    # Estadísticas
    media = df[var].mean()
    axes[i].axhline(media, color='blue', linestyle=':', linewidth=1, alpha=0.5)

plt.suptitle("Boxplots de Variables - AgroExport Ica", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "eda_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Guardado: graficos/eda_boxplots.png")

# %% Celda 7: Análisis de relaciones no lineales
print("\n📊 Análisis de relaciones no lineales...")
print("=" * 60)

# Variables a analizar contra rendimiento
vars_analizar = ['litros_riego', 'kg_fertilizante', 'antiguedad_terreno', 'radiacion_solar_acumulada']
vars_analizar = [v for v in vars_analizar if v in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(vars_analizar):
    ax = axes[i]

    # Normalizar variable por hectárea si tiene sentido
    if var == 'litros_riego':
        x = df[var] / df['hectareas']
        xlabel = f"{var} / hectárea"
    elif var == 'kg_fertilizante':
        x = df[var] / df['hectareas']
        xlabel = f"{var} / hectárea"
    else:
        x = df[var]
        xlabel = var

    y = df['toneladas'] / df['hectareas']  # rendimiento por hectárea

    ax.scatter(x, y, alpha=0.3, s=15, color='steelblue')

    # Ajuste polinómico de grado 2
    try:
        z2 = np.polyfit(x, y, 2)
        p2 = np.poly1d(z2)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, p2(x_sorted), 'r-', linewidth=2, label='Ajuste cuadrático')
    except Exception:
        pass

    # Ajuste polinómico de grado 3
    try:
        z3 = np.polyfit(x, y, 3)
        p3 = np.poly1d(z3)
        ax.plot(x_sorted, p3(x_sorted), 'g--', linewidth=1.5, label='Ajuste cúbico')
    except Exception:
        pass

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Rendimiento (t/ha)", fontsize=10)
    ax.set_title(f"Rendimiento vs {var}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(22, color='gray', linestyle=':', alpha=0.5, label='Objetivo sintético')

plt.suptitle(
    "Análisis de Relaciones No Lineales con el Rendimiento",
    fontsize=14, fontweight='bold',
)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "eda_relaciones_no_lineales.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Guardado: graficos/eda_relaciones_no_lineales.png")

print("\n" + "=" * 60)
print("✅ Análisis Exploratorio de Datos completado")
print(f"📁 Gráficos guardados en: {GRAFICOS_DIR}")
print("=" * 60)
