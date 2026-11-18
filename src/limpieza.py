"""
limpieza.py - Limpieza y validación del dataset de cosechas de Ica
==================================================================
Proyecto: AgroExport Ica - Predicción de cosecha de uva de mesa

Este script:
1. Lee el CSV generado por generar_dataset.py
2. Verifica tipos de datos y valores nulos
3. Detecta y marca outliers usando el método IQR (no los elimina)
4. Verifica rangos y unidades esperadas
5. Genera un reporte completo de limpieza
6. Guarda el dataset limpio con columna de outliers marcados
"""

import numpy as np
import pandas as pd
import os
import sys

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "historial_cosechas_ica.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "historial_cosechas_ica_limpio.csv")

# Rangos esperados por variable (para validación de unidades)
RANGOS_ESPERADOS = {
    "hectareas": {
        "min": 2.0, "max": 50.0, "unidad": "hectáreas",
        "descripcion": "Rango simulado para el prototipo; no son registros de fundos reales"
    },
    "litros_riego": {
        "min": 10_000_000, "max": 600_000_000, "unidad": "litros totales por lote",
        "descripcion": "Supuesto simulado: 5,000-12,000 m³/ha convertido a litros totales"
    },
    "radiacion_solar_acumulada": {
        "min": 1238.3, "max": 1340.4, "unidad": "kWh/m² por campaña (octubre-marzo)",
        "descripcion": "NASA POWER diario agregado para campañas octubre-marzo; MJ convertido a kWh"
    },
    "kg_fertilizante": {
        "min": 300.0, "max": 30000.0, "unidad": "kg totales por lote",
        "descripcion": "Supuesto simulado: 150-600 kg/ha multiplicado por hectáreas"
    },
    "antiguedad_terreno": {
        "min": 1, "max": 30, "unidad": "años",
        "descripcion": "Edad del viñedo (1-30 años)"
    },
    "toneladas": {
        "min": 10.0, "max": 2000.0, "unidad": "toneladas totales del lote",
        "descripcion": "Objetivo semi-sintético: rendimiento medio ~22 t/ha × hectáreas"
    }
}

# Tipos de datos esperados
TIPOS_ESPERADOS = {
    "lote_id": "object",
    "hectareas": "float64",
    "litros_riego": "int64",
    "radiacion_solar_acumulada": "float64",
    "kg_fertilizante": "float64",
    "antiguedad_terreno": "int64",
    "toneladas": "float64"
}


def cargar_dataset(filepath):
    """Carga el dataset CSV y retorna el DataFrame."""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: No se encontró el archivo: {filepath}")
        print("   Ejecuta primero: python src/generar_dataset.py")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {filepath}")
    print(f"   Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def verificar_tipos(df):
    """Verifica que los tipos de datos coincidan con los esperados."""
    print("\n" + "=" * 70)
    print("1. VERIFICACIÓN DE TIPOS DE DATOS")
    print("=" * 70)

    errores = 0
    for col, tipo_esperado in TIPOS_ESPERADOS.items():
        tipo_actual = str(df[col].dtype)
        # pandas 2.x usa 'string' en vez de 'object' para texto
        ok = (tipo_actual == tipo_esperado or
              (tipo_esperado == "object" and tipo_actual in ("object", "str", "string")))
        status = "✅" if ok else "⚠️"
        print(f"   {status} {col:30s} | Esperado: {tipo_esperado:10s} | Actual: {tipo_actual:10s}")
        if not ok:
            errores += 1

    if errores == 0:
        print("\n   ✅ Todos los tipos de datos son correctos.")
    else:
        print(f"\n   ⚠️  {errores} tipo(s) de dato no coinciden.")

    return errores


def verificar_nulos(df):
    """Verifica valores nulos en cada columna."""
    print("\n" + "=" * 70)
    print("2. VERIFICACIÓN DE VALORES NULOS")
    print("=" * 70)

    nulos_por_col = df.isnull().sum()
    total_nulos = nulos_por_col.sum()

    for col in df.columns:
        n = nulos_por_col[col]
        pct = (n / len(df)) * 100
        status = "✅" if n == 0 else "⚠️"
        print(f"   {status} {col:30s} | Nulos: {n:4d} ({pct:.1f}%)")

    print(f"\n   Total de valores nulos: {total_nulos}")
    if total_nulos == 0:
        print("   ✅ No hay valores nulos en el dataset.")
    else:
        print(f"   ⚠️  Se encontraron {total_nulos} valores nulos.")

    return total_nulos


def verificar_duplicados(df):
    """Verifica registros duplicados."""
    print("\n" + "=" * 70)
    print("3. VERIFICACIÓN DE DUPLICADOS")
    print("=" * 70)

    # Duplicados en lote_id
    dup_id = df["lote_id"].duplicated().sum()
    print(f"   lote_id duplicados: {dup_id} {'✅' if dup_id == 0 else '⚠️'}")

    # Duplicados en todas las columnas
    dup_total = df.duplicated().sum()
    print(f"   Filas completamente duplicadas: {dup_total} {'✅' if dup_total == 0 else '⚠️'}")

    return dup_id + dup_total


def verificar_unidades_rangos(df):
    """Verifica que los valores estén dentro de los rangos esperados por unidad."""
    print("\n" + "=" * 70)
    print("4. VERIFICACIÓN DE UNIDADES Y RANGOS")
    print("=" * 70)

    problemas = 0
    for col, info in RANGOS_ESPERADOS.items():
        val_min = df[col].min()
        val_max = df[col].max()
        en_rango_min = val_min >= info["min"] * 0.5  # margen del 50%
        en_rango_max = val_max <= info["max"] * 1.5  # margen del 50%
        ok = en_rango_min and en_rango_max
        status = "✅" if ok else "⚠️"

        print(f"\n   {status} {col}")
        print(f"      Unidad: {info['unidad']}")
        print(f"      Rango esperado: [{info['min']:,.1f}, {info['max']:,.1f}]")
        print(f"      Rango actual:   [{val_min:,.1f}, {val_max:,.1f}]")
        print(f"      Fuente: {info['descripcion']}")

        if not ok:
            problemas += 1

    # Verificación especial: rendimiento por hectárea
    rend_tha = df["toneladas"] / df["hectareas"]
    rend_medio = rend_tha.mean()
    print(f"\n   📊 Rendimiento medio semi-sintético: {rend_medio:.2f} t/ha")
    print("      Objetivo de generación: ~22 t/ha; no es una cifra oficial de MIDAGRI")
    if 20 <= rend_medio <= 25:
        print(f"      ✅ Dentro del objetivo de calibración del prototipo")
    else:
        print(f"      ⚠️  Fuera del objetivo de calibración del prototipo")
        problemas += 1

    return problemas


def detectar_outliers_iqr(df, columnas_numericas):
    """
    Detecta outliers usando el método IQR (Rango Intercuartílico).
    NO elimina outliers, solo los marca con una columna booleana.

    Método IQR:
    - Q1 = percentil 25, Q3 = percentil 75
    - IQR = Q3 - Q1
    - Outlier si: valor < Q1 - 1.5*IQR  OR  valor > Q3 + 1.5*IQR

    Retorna DataFrame con columna 'es_outlier' añadida.
    """
    print("\n" + "=" * 70)
    print("5. DETECCIÓN DE OUTLIERS (Método IQR)")
    print("=" * 70)
    print("   Criterio: Outlier si valor < Q1 - 1.5×IQR o valor > Q3 + 1.5×IQR")
    print("   NOTA: Los outliers se MARCAN pero NO se eliminan.")

    df = df.copy()
    df["es_outlier"] = False
    outlier_detalle = {}

    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR

        outliers_col = (df[col] < limite_inf) | (df[col] > limite_sup)
        n_outliers = outliers_col.sum()
        pct = (n_outliers / len(df)) * 100

        # Marcar en la columna general
        df.loc[outliers_col, "es_outlier"] = True

        outlier_detalle[col] = {
            "n": n_outliers, "pct": pct,
            "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "limite_inf": limite_inf, "limite_sup": limite_sup
        }

        status = "✅" if n_outliers == 0 else "⚠️"
        print(f"\n   {status} {col}:")
        print(f"      Q1={Q1:,.2f}  Q3={Q3:,.2f}  IQR={IQR:,.2f}")
        print(f"      Límites: [{limite_inf:,.2f}, {limite_sup:,.2f}]")
        print(f"      Outliers: {n_outliers} ({pct:.1f}%)")

    total_outliers = df["es_outlier"].sum()
    pct_total = (total_outliers / len(df)) * 100
    print(f"\n   📊 Total de lotes marcados como outlier: {total_outliers} ({pct_total:.1f}%)")

    return df, outlier_detalle


def generar_reporte_resumen(df, n_nulos, n_dup, n_tipo_err, n_rango_prob, outlier_detalle):
    """Genera un reporte resumen de toda la limpieza."""
    print("\n" + "=" * 70)
    print("📋 REPORTE RESUMEN DE LIMPIEZA")
    print("=" * 70)

    total_outliers = df["es_outlier"].sum() if "es_outlier" in df.columns else 0

    print(f"""
   Dataset: historial_cosechas_ica.csv
   Registros: {len(df)}
   Columnas: {len(df.columns)}

   ┌──────────────────────────────────┬──────────┐
   │ Verificación                     │ Estado   │
   ├──────────────────────────────────┼──────────┤
   │ Tipos de datos correctos         │ {'✅ OK' if n_tipo_err == 0 else f'⚠️  {n_tipo_err} err':8s} │
   │ Valores nulos                    │ {'✅ 0' if n_nulos == 0 else f'⚠️  {n_nulos}':8s} │
   │ Registros duplicados             │ {'✅ 0' if n_dup == 0 else f'⚠️  {n_dup}':8s} │
   │ Rangos y unidades válidos        │ {'✅ OK' if n_rango_prob == 0 else f'⚠️  {n_rango_prob} prob':8s} │
   │ Outliers detectados (IQR)        │ {total_outliers:>4d}     │
   └──────────────────────────────────┴──────────┘
    """)

    # Rendimiento final
    rend = df["toneladas"] / df["hectareas"]
    print(f"   Rendimiento medio final: {rend.mean():.2f} ± {rend.std():.2f} t/ha")
    print("   El objetivo sintético de 22 t/ha no es una media oficial.")

    # Estadísticas descriptivas
    print("\n   📊 Estadísticas descriptivas:")
    cols_num = ["hectareas", "litros_riego", "radiacion_solar_acumulada",
                "kg_fertilizante", "antiguedad_terreno", "toneladas"]
    print(df[cols_num].describe().round(2).to_string(col_space=15))


def limpiar_dataset():
    """Función principal de limpieza del dataset."""
    print("🍇 AgroExport Ica - Limpieza y Validación del Dataset")
    print("=" * 70)

    # 1. Cargar datos
    df = cargar_dataset(INPUT_FILE)

    # 2. Verificar tipos
    n_tipo_err = verificar_tipos(df)

    # 3. Verificar nulos
    n_nulos = verificar_nulos(df)

    # 4. Verificar duplicados
    n_dup = verificar_duplicados(df)

    # 5. Verificar unidades y rangos
    n_rango_prob = verificar_unidades_rangos(df)

    # 6. Detectar outliers con IQR
    columnas_numericas = ["hectareas", "litros_riego", "radiacion_solar_acumulada",
                          "kg_fertilizante", "antiguedad_terreno", "toneladas"]
    df, outlier_detalle = detectar_outliers_iqr(df, columnas_numericas)

    # 7. Reporte resumen
    generar_reporte_resumen(df, n_nulos, n_dup, n_tipo_err, n_rango_prob, outlier_detalle)

    # 8. Guardar dataset limpio (con columna de outliers marcados)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Dataset limpio guardado en: {OUTPUT_FILE}")
    print(f"   Columna 'es_outlier' añadida (True = outlier detectado por IQR)")
    print(f"   Los outliers están MARCADOS pero NO eliminados.")
    print(f"   Registros: {len(df)} | Tamaño: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    return df


if __name__ == "__main__":
    df = limpiar_dataset()
