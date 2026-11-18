"""Build a traceable semi-synthetic grape-harvest dataset for Ica."""

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
N_LOTES = 400
RANDOM_STATE = 42
NASA_SOURCE_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude=-75.73"
    "&latitude=-14.07&start=20150101&end=20241231&format=JSON"
)

MIDAGRI_WORKBOOKS = {
    2016: (
        "midagri_anuario_produccion_agricola_2016.xlsx",
        "https://cdn.www.gob.pe/uploads/document/file/2803230/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202016.xlsx"
        "?v=1772635690",
    ),
    2017: (
        "midagri_anuario_produccion_agricola_2017.xls",
        "https://cdn.www.gob.pe/uploads/document/file/2803228/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202017.xls"
        "?v=1772635693",
    ),
    2018: (
        "midagri_anuario_produccion_agricola_2018.xls",
        "https://cdn.www.gob.pe/uploads/document/file/2803226/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202018.xls"
        "?v=1772635710",
    ),
    2019: (
        "midagri_anuario_produccion_agricola_2019.xls",
        "https://cdn.www.gob.pe/uploads/document/file/3754199/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202019.xls"
        "?v=1772635716",
    ),
    2020: (
        "midagri_anuario_produccion_agricola_2020.xls",
        "https://cdn.www.gob.pe/uploads/document/file/3748650/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202020.xls"
        "?v=1772635723",
    ),
    2021: (
        "midagri_anuario_produccion_agricola_2021.xls",
        "https://cdn.www.gob.pe/uploads/document/file/3748735/"
        "Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202021.xls"
        "?v=1772635723",
    ),
    2022: (
        "midagri_anuario_produccion_agricola_2022.xlsx",
        "https://cdn.www.gob.pe/uploads/document/file/6770784/"
        "2730325-cuadros-en-excel-del-anuario-produccion-agricola-2022%282%29.xlsx"
        "?v=1772635737",
    ),
    2023: (
        "midagri_anuario_produccion_agricola_2023.xlsx",
        "https://cdn.www.gob.pe/uploads/document/file/9547451/"
        "2730325-cuadros-en-excel-del-anuario-produccion-agricola-2023.xlsx"
        "?v=1772635737",
    ),
}


def _normalize(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text if not unicodedata.combining(char)).strip().lower()


def _number(value, label, year):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valor MIDAGRI no numérico para {label}, año {year}: {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"Valor MIDAGRI inválido para {label}, año {year}: {value!r}")
    return number


def extract_midagri_series():
    """Extract the Ica grape summary row from each downloaded annual workbook."""
    records = []
    for year, (filename, source_url) in MIDAGRI_WORKBOOKS.items():
        path = RAW_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Falta el archivo oficial de MIDAGRI: {path}\n"
                f"Descárgalo desde: {source_url}"
            )

        workbook = pd.ExcelFile(path)
        sheet = next(
            (name for name in workbook.sheet_names if _normalize(name) == "uva"),
            None,
        )
        if sheet is None and year == 2016:
            sheet = next(
                (name for name in workbook.sheet_names if _normalize(name) == "c.98"),
                None,
            )
        if sheet is None:
            raise ValueError(f"No se encontró la hoja 'uva' en {path.name}")
        frame = pd.read_excel(workbook, sheet_name=sheet, header=None)

        if year == 2016:
            header_row, region_col, area_col, production_col, yield_col = (
                2,
                0,
                2,
                6,
                10,
            )
        else:
            indices = None
            for row_index, row in frame.iterrows():
                cells = [_normalize(value) for value in row.tolist()]
                region_index = next(
                    (i for i, value in enumerate(cells) if value == "region"),
                    None,
                )
                production_index = next(
                    (i for i, value in enumerate(cells) if "produccion" in value),
                    None,
                )
                area_index = next(
                    (i for i, value in enumerate(cells) if "superficie" in value),
                    None,
                )
                yield_index = next(
                    (i for i, value in enumerate(cells) if "rendimiento" in value),
                    None,
                )
                if None not in (
                    region_index,
                    production_index,
                    area_index,
                    yield_index,
                ):
                    indices = (
                        row_index,
                        region_index,
                        production_index,
                        area_index,
                        yield_index,
                    )
                    break

            if indices is None:
                raise ValueError(f"No se encontró la tabla regional de uva en {path.name}")

            header_row, region_col, production_col, area_col, yield_col = indices
        ica_row = None
        for _, row in frame.iloc[header_row + 1 :].iterrows():
            if _normalize(row.iloc[region_col]) == "ica":
                ica_row = row
                break
        if ica_row is None:
            raise ValueError(f"No se encontró la fila de Ica en la hoja 'uva' de {path.name}")

        production_t = _number(ica_row.iloc[production_col], "producción (t)", year)
        area_ha = _number(ica_row.iloc[area_col], "superficie (ha)", year)
        reported_yield_kg_ha = _number(ica_row.iloc[yield_col], "rendimiento (kg/ha)", year)
        if area_ha <= 0:
            raise ValueError(f"Superficie cosechada MIDAGRI no positiva para Ica, año {year}")

        records.append(
            {
                "anio": year,
                "produccion_t": production_t,
                "superficie_cosechada_ha": area_ha,
                "rendimiento_reportado_kg_ha": reported_yield_kg_ha,
                "rendimiento_calculado_t_ha": production_t / area_ha,
                "url_fuente": source_url,
            }
        )

    series = pd.DataFrame(records).sort_values("anio")
    output = RAW_DIR / "midagri_uva_ica_2016_2023.csv"
    series.to_csv(output, index=False, encoding="utf-8")
    return series


def load_campaign_radiation():
    """Aggregate observed NASA daily radiation into complete Oct-Mar campaigns."""
    path = RAW_DIR / "nasa_power_ica_2015_2024.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta el archivo NASA POWER: {path}\n"
            f"Descárgalo desde: {NASA_SOURCE_URL}"
        )

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    try:
        daily = payload["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    except (KeyError, TypeError) as exc:
        raise ValueError("El JSON NASA POWER no contiene ALLSKY_SFC_SW_DWN") from exc

    dates = pd.to_datetime(list(daily.keys()), format="%Y%m%d", errors="raise")
    values = pd.to_numeric(pd.Series(list(daily.values())), errors="coerce")
    observations = pd.DataFrame({"fecha": dates, "radiacion_mj_m2_dia": values})
    valid = observations["radiacion_mj_m2_dia"].notna()
    valid &= observations["radiacion_mj_m2_dia"] >= 0
    observations = observations.loc[valid].set_index("fecha")

    campaigns = []
    for start_year in range(2015, 2024):
        start = pd.Timestamp(start_year, 10, 1)
        end = pd.Timestamp(start_year + 1, 3, 31)
        expected_dates = pd.date_range(start, end, freq="D")
        observed = observations.reindex(expected_dates)["radiacion_mj_m2_dia"]
        if observed.isna().any():
            missing = observed[observed.isna()].index.strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"NASA POWER tiene {len(missing)} días faltantes en "
                f"{start_year}-{start_year + 1}; no se imputarán."
            )

        campaigns.append(
            {
                "campania": f"{start_year}-{start_year + 1}",
                "radiacion_solar_acumulada_kwh_m2": observed.sum() / 3.6,
                "dias_observados": len(observed),
                "unidad_origen": "MJ/m²/día",
                "url_fuente": NASA_SOURCE_URL,
            }
        )

    campaign_frame = pd.DataFrame(campaigns)
    campaign_frame.to_csv(
        RAW_DIR / "nasa_power_radiacion_campanas_2015_2024.csv",
        index=False,
        encoding="utf-8",
    )
    return campaign_frame


def efecto_riego(riego_m3_ha):
    optimo_m3_ha = 8500.0
    sigma_m3_ha = 2500.0
    return np.exp(-0.5 * ((riego_m3_ha - optimo_m3_ha) / sigma_m3_ha) ** 2)


def efecto_fertilizante(kg_fert_ha):
    return np.log1p(kg_fert_ha) / np.log1p(600.0)


def efecto_antiguedad(anos):
    resultado = np.zeros_like(anos, dtype=float)
    joven = anos <= 8
    meseta = (anos > 8) & (anos <= 18)
    viejo = anos > 18
    resultado[joven] = 0.4 + 0.6 * (1 - np.exp(-0.5 * anos[joven]))
    resultado[meseta] = 1.0
    resultado[viejo] = 1.0 - 0.035 * (anos[viejo] - 18) ** 1.5
    return np.clip(resultado, 0.15, 1.0)


def efecto_radiacion(radiacion_kwh_m2, mediana_kwh_m2):
    return 0.7 + 0.3 / (
        1 + np.exp(-0.02 * (radiacion_kwh_m2 - mediana_kwh_m2))
    )


def _calibrate_yield(raw_yield, target_mean):
    """Scale and clip a noisy yield curve to the user-specified dataset mean."""
    lower, upper = 0.0, max(1.0, target_mean / raw_yield.mean() * 4)
    while np.clip(upper * raw_yield, 5.0, 40.0).mean() < target_mean:
        upper *= 2
    for _ in range(80):
        scale = (lower + upper) / 2
        if np.clip(scale * raw_yield, 5.0, 40.0).mean() < target_mean:
            lower = scale
        else:
            upper = scale
    return np.clip(((lower + upper) / 2) * raw_yield, 5.0, 40.0)


def generar_dataset(n_lotes=N_LOTES, random_state=RANDOM_STATE):
    if n_lotes <= 0:
        raise ValueError("n_lotes debe ser mayor que cero")

    series_midagri = extract_midagri_series()
    campaigns = load_campaign_radiation()
    target_mean_t_ha = series_midagri["rendimiento_calculado_t_ha"].mean()
    rng = np.random.default_rng(random_state)

    campaign_names = campaigns["campania"].to_numpy()
    selected_campaigns = rng.choice(campaign_names, size=n_lotes, replace=True)
    radiation_by_campaign = campaigns.set_index("campania")[
        "radiacion_solar_acumulada_kwh_m2"
    ]
    radiation = np.array([radiation_by_campaign[name] for name in selected_campaigns])

    hectares = rng.uniform(2, 50, size=n_lotes)
    irrigation_m3_ha = rng.uniform(5000, 12000, size=n_lotes)
    fertilizer_kg_ha = rng.uniform(150, 600, size=n_lotes)
    age = rng.integers(1, 31, size=n_lotes).astype(float)

    raw_yield = (
        efecto_riego(irrigation_m3_ha)
        * efecto_fertilizante(fertilizer_kg_ha)
        * efecto_antiguedad(age)
        * efecto_radiacion(radiation, campaigns[
            "radiacion_solar_acumulada_kwh_m2"
        ].median())
        * rng.normal(1.0, 0.12, size=n_lotes)
    )
    yield_t_ha = _calibrate_yield(raw_yield, target_mean_t_ha)

    dataset = pd.DataFrame(
        {
            "lote_id": [f"LOTE-{index:04d}" for index in range(1, n_lotes + 1)],
            "campania": selected_campaigns,
            "hectareas": hectares,
            "litros_riego": irrigation_m3_ha * 1000 * hectares,
            "radiacion_solar_acumulada": radiation,
            "kg_fertilizante": fertilizer_kg_ha * hectares,
            "antiguedad_terreno": age.astype(int),
            "toneladas": yield_t_ha * hectares,
        }
    )

    dataset["hectareas"] = dataset["hectareas"].round(2)
    dataset["litros_riego"] = dataset["litros_riego"].round().astype("int64")
    dataset["radiacion_solar_acumulada"] = dataset["radiacion_solar_acumulada"].round(2)
    dataset["kg_fertilizante"] = dataset["kg_fertilizante"].round(2)
    dataset["toneladas"] = dataset["toneladas"].round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(
        OUTPUT_DIR / "historial_cosechas_ica.csv",
        index=False,
        encoding="utf-8",
    )

    mean_midagri = series_midagri["rendimiento_calculado_t_ha"].mean()
    std_midagri = series_midagri["rendimiento_calculado_t_ha"].std(ddof=1)
    yield_per_ha = dataset["toneladas"] / dataset["hectareas"]

    print("Rendimiento MIDAGRI Ica (media simple 2016-2023):")
    print(f"  media = {mean_midagri:.2f} t/ha; desviación estándar = {std_midagri:.2f} t/ha")
    print("Rendimiento semi-sintético generado (calibrado con la media anual MIDAGRI):")
    print(f"  media = {yield_per_ha.mean():.2f} t/ha; desviación estándar = {yield_per_ha.std():.2f} t/ha")
    print("Radiación NASA POWER por campaña (kWh/m² acumulados):")
    print(campaigns[["campania", "radiacion_solar_acumulada_kwh_m2"]].to_string(index=False))
    print(f"CSV creado: {OUTPUT_DIR / 'historial_cosechas_ica.csv'} ({len(dataset)} lotes)")
    return dataset


if __name__ == "__main__":
    generar_dataset()
