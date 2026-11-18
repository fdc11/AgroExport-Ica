# Fuentes y trazabilidad — AgroExport Ica

Fecha de consulta y descarga: 18 de noviembre de 2026.

## Estado de las fuentes

| Variable / dato | Fuente (URL exacta) | Tipo | Cómo se obtuvo |
|---|---|---|---|
| Radiación solar diaria | [NASA POWER API](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude=-75.73&latitude=-14.07&start=20150101&end=20241231&format=JSON) | Real | Respuesta guardada en `data/raw/nasa_power_ica_2015_2024.json`, para -14.07, -75.73, del 01-01-2015 al 31-12-2024. La cabecera CSV oficial identifica `ALLSKY_SFC_SW_DWN` como MJ/m²/día; ver [metadatos del parámetro](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude=-75.73&latitude=-14.07&start=20150101&end=20150102&format=CSV). |
| Radiación acumulada por campaña | NASA POWER, misma URL de arriba | Real, agregada | Suposición del prototipo: campaña octubre-marzo. Se suman los valores diarios de cada periodo completo; la unidad diaria de NASA se convierte de MJ/m²/día a kWh/m²/día dividiendo entre 3.6. El total se guarda en `data/raw/nasa_power_radiacion_campanas_2015_2024.csv`. Hay nueve campañas completas: 2015-2016 a 2023-2024. |
| Producción de uva en Ica | [Compendio anual de Producción Agrícola MIDAGRI/SIEA](https://www.gob.pe/institucion/midagri/informes-publicaciones/2730325-compendio-anual-de-produccion-agricola) y los XLS/XLSX anuales enlazados abajo | Real, regional y anual | Se extraen las filas de Ica de las hojas de uva en los anuarios descargados (2016-2023); archivos originales en `data/raw/`. |
| Superficie cosechada de uva en Ica | Mismos anuarios MIDAGRI/SIEA | Real, regional y anual | Valor reportado por el anuario, en hectáreas. |
| Rendimiento anual de uva en Ica | Mismos anuarios MIDAGRI/SIEA | Real, regional y anual | Se conserva el rendimiento publicado en kg/ha y se calcula además t/ha como producción (t) / superficie (ha). La serie extraída está en `data/raw/midagri_uva_ica_2016_2023.csv`. |
| Rendimiento nacional de uva | [FAOSTAT, Crops and livestock products (QCL)](https://www.fao.org/faostat/en/#data/QCL); intento de API: `https://fenixservices.fao.org/faostat/api/v1/en/data/QCL?area_code=170&item_code=560&element_code=5419&year=2024` | No descargado | El intento al API de FAOSTAT devolvió HTTP 521. No se usaron ni sustituyeron cifras FAOSTAT. Por tanto, las estadísticas de calibración disponibles son solo las de MIDAGRI/SIEA. |
| Hectáreas por lote | No hay registros de lote disponibles en las fuentes descargadas | Simulada | Supuesto del prototipo: uniforme entre 2 y 50 ha. No representa observaciones reales de AgroExport. |
| Riego | No se descargó una serie primaria de riego por lote/campaña | Simulada | Supuesto del prototipo: 5,000-12,000 m³/ha/campaña, convertido a litros totales (`m³/ha × 1,000 × ha`). El punto óptimo de 8,500 m³/ha y la curva gaussiana también son supuestos de modelado, no mediciones de ANA. |
| Fertilizante | No hay registros de fertilización por lote disponibles | Simulada | Supuesto del prototipo: 150-600 kg/ha, multiplicado por hectáreas. El efecto logarítmico de rendimientos decrecientes es una hipótesis del generador. |
| Antigüedad del viñedo | No hay registros de edad por lote disponibles | Simulada | Entero uniforme de 1 a 30 años. La curva de maduración, meseta y declive es una hipótesis del generador. |
| Toneladas por lote | No existen cosechas históricas reales por lote en este proyecto | Semi-sintética | Rendimiento construido con efectos no lineales de riego, fertilizante, edad y radiación, más ruido gaussiano multiplicativo (desviación 12%). El factor de escala se ajusta para que el promedio simulado iguale la media simple de los rendimientos anuales de MIDAGRI/SIEA para Ica (2016-2023; ~22 t/ha). Se limita el rendimiento simulado a 5-40 t/ha. Toneladas = rendimiento × hectáreas. |

## Archivos anuales descargados de MIDAGRI/SIEA

Cada enlace de descarga procede del compendio MIDAGRI enlazado arriba. La serie que se extrajo contiene los años disponibles 2016-2023; no se encontraron en esta descarga anuarios 2015 ni 2024.

| Año | Archivo local | URL exacta de descarga |
|---:|---|---|
| 2016 | `data/raw/midagri_anuario_produccion_agricola_2016.xlsx` | [XLSX 2016](https://cdn.www.gob.pe/uploads/document/file/2803230/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202016.xlsx?v=1772635690) |
| 2017 | `data/raw/midagri_anuario_produccion_agricola_2017.xls` | [XLS 2017](https://cdn.www.gob.pe/uploads/document/file/2803228/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202017.xls?v=1772635693) |
| 2018 | `data/raw/midagri_anuario_produccion_agricola_2018.xls` | [XLS 2018](https://cdn.www.gob.pe/uploads/document/file/2803226/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202018.xls?v=1772635710) |
| 2019 | `data/raw/midagri_anuario_produccion_agricola_2019.xls` | [XLS 2019](https://cdn.www.gob.pe/uploads/document/file/3754199/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202019.xls?v=1772635716) |
| 2020 | `data/raw/midagri_anuario_produccion_agricola_2020.xls` | [XLS 2020](https://cdn.www.gob.pe/uploads/document/file/3748650/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202020.xls?v=1772635723) |
| 2021 | `data/raw/midagri_anuario_produccion_agricola_2021.xls` | [XLS 2021](https://cdn.www.gob.pe/uploads/document/file/3748735/Cuadros%20en%20Excel%20del%20anuario%20%22PRODUCCI%C3%93N%20AGR%C3%8DCOLA%22%202021.xls?v=1772635723) |
| 2022 | `data/raw/midagri_anuario_produccion_agricola_2022.xlsx` | [XLSX 2022](https://cdn.www.gob.pe/uploads/document/file/6770784/2730325-cuadros-en-excel-del-anuario-produccion-agricola-2022%282%29.xlsx?v=1772635737) |
| 2023 | `data/raw/midagri_anuario_produccion_agricola_2023.xlsx` | [XLSX 2023](https://cdn.www.gob.pe/uploads/document/file/9547451/2730325-cuadros-en-excel-del-anuario-produccion-agricola-2023.xlsx?v=1772635737) |

## Estadística oficial de calibración

El generador extrae para cada año la producción, superficie cosechada y rendimiento que reporta MIDAGRI/SIEA para Ica. La media y la desviación estándar impresas al ejecutar el script corresponden a los rendimientos anuales de 2016-2023 (media simple, desviación estándar muestral); no son estadísticas de 400 lotes ni incluyen FAOSTAT. El objetivo de media del dataset semi-sintético se deriva de la media observada MIDAGRI, no es un valor oficial por lote.

## Límites de interpretación

- Solo la radiación procede de datos diarios observados y descargados de NASA POWER. Al asignar campañas a 400 lotes, los nueve totales observados se reutilizan; no son 400 mediciones independientes.
- MIDAGRI aporta estadísticas agregadas por región y año, no cosechas de lotes de esta empresa. Se usan como referencia y no como filas de entrenamiento.
- Los registros de lotes, insumos y toneladas son simulados. Una métrica de validación sobre este CSV mide qué tan bien el modelo aprende la función artificial usada para generarlo; no estima precisión en fundos reales.
- Los valores asumidos para riego, fertilizante, edad, efectos agronómicos y ruido deben sustituirse por registros de campo antes de usar el modelo en planificación operativa.
