# Sura Flows Dashboard

Dashboard Streamlit para analizar flujos AFP y FFMM vs IPSA a partir del archivo `Request_Sura.xlsx`.

## Archivos

- `sura_pipeline.py` — Pipeline de carga, features, etiquetas y señales (AFP y FFMM procesados en funciones totalmente separadas).
- `app.py` — UI Streamlit con Panorama Ejecutivo + 11 tabs + toggle global + export Excel.
- `requirements.txt` — dependencias.

## Estructura

```
tu-carpeta/
├── sura_pipeline.py
├── app.py
├── requirements.txt
└── Request_Sura.xlsx   (tu archivo mensual, no incluido)
```

-----

## Instalación y uso local

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate           # Linux / Mac
# o
venv\Scripts\activate              # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Correr el dashboard

```bash
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

### 4. Cargar el Excel

- En el sidebar, subí `Request_Sura.xlsx` o pegá la ruta local.
- Presioná **Cargar y ejecutar**.

-----

## Deploy en Streamlit Cloud

Para replicar el dashboard anterior en `streamlit.app`:

1. Subí los 3 archivos (`sura_pipeline.py`, `app.py`, `requirements.txt`) a un repo de GitHub.
1. En [share.streamlit.io](https://share.streamlit.io/) conectá el repo.
1. Main file: `app.py`.
1. Deploy.

-----

## Estructura del dashboard

### Panorama Ejecutivo (landing, siempre visible)

Tabla tipo research con Ticker, Peso IPSA, Peso AFP, Diff AFP en bps (barra verde/roja), Peso FFMM, Diff FFMM en bps (barra verde/roja). Ordenada por peso IPSA descendente.

### Sidebar persistente

- Uploader de Excel.
- **Toggle global Universo**: AFP / FFMM / Ambos.
- Resumen ejecutivo dinámico (se adapta al toggle).
- Botón Export Excel (snapshot + events).

### Tabs analíticas

|# |Nombre                       |Respeta toggle                          |
|--|-----------------------------|----------------------------------------|
|1 |📈 Posicionamiento vs historia|Sí                                      |
|2 |✅ Snapshot                   |Sí                                      |
|3 |🏁 Ranking                    |Sí                                      |
|4 |📊 Detalle por papel          |No (siempre AFP y FFMM)                 |
|5 |🟦 Heatmap                    |Sí (en “Ambos” → heatmap de divergencia)|
|6 |📊 Flujo 1M/3M/6M             |Sí                                      |
|7 |🌐 Breadth                    |No (siempre ambos)                      |
|8 |🎯 Scatter AFP vs FFMM        |No (siempre ambos)                      |
|9 |⚡ Liderazgo                  |No (siempre ambos)                      |
|10|🏢 Sectorial                  |Sí                                      |
|11|🔄 Persistencia               |Sí                                      |

-----

## Sistema de etiquetas

AFP y FFMM tienen dos ejes ortogonales independientes:

**Posicionamiento** (dónde está parado el fondo):

- `OW_Extremo`: GAP > 0 y Z-score ≥ +1.5
- `OW`: GAP > 0
- `Neutral`: |GAP| < 5 bps
- `UW`: GAP < 0
- `UW_Extremo`: GAP < 0 y Z-score ≤ −1.5

**Dirección** (qué está haciendo):

- `Acumulando_Fuerte`: ΔGAP > 0 + persistencia ≥ 3M + umbral adaptativo
- `Acumulando`: ΔGAP > 0
- `Plano`: |ΔGAP| < 5 bps
- `Reduciendo`: ΔGAP < 0
- `Reduciendo_Fuerte`: ΔGAP < 0 + persistencia ≥ 3M + umbral adaptativo

**Umbral adaptativo “Fuerte”** (ambos criterios):

1. `|ΔGAP| > max(5 bps, 0.5·σ_ticker)` del propio ticker.
1. `|ΔGAP| ≥ percentil 85` histórico del propio ticker.

**Señal** sale de matriz explícita Posicionamiento × Dirección (ver expander en Tab 2 Snapshot).

**Persistencia**: `+4M` = 4 meses seguidos comprando, `-3M` = 3 meses seguidos vendiendo.

-----

## Cruce AFP ↔ FFMM

Se calcula **después** de los pipelines independientes, sin modificarlos:

- `Divergencia_GAP = GAP_AFP − GAP_FFMM` con z-score 6M.
- `Corr_6M`: correlación rolling entre ΔGAP_AFP y ΔGAP_FFMM.
- `Lead_Lag`: lag óptimo de correlación cruzada (-3 a +3). Negativo = AFP lidera, positivo = FFMM lidera.
- `Liderazgo_del_mes`: Lidera_AFP / Lidera_FFMM / Consenso_Compra / Consenso_Venta / Divergencia_Flujos / Sin_Señal.

-----

## Actualización mensual

1. Reemplazá `Request_Sura.xlsx` en la carpeta (o subilo al uploader).
1. Presioná **Cargar y ejecutar**.

El parser detecta automáticamente nuevas columnas de fecha (`YYYY_M`) sin intervención manual.

-----

## Troubleshooting

**“No se detectaron todos los bloques”**: el parser espera que la hoja `Weights` tenga los 6 marcadores en columna A (`IPSA`, `IPSA - Weights`, `Pension Funds`, `Pension Funds - Weight`, `Mutual Funds`, `Mutual Funds - Weight`). Verificá que no se hayan renombrado.

**Pocas señales BUY/SELL**: es esperable. Los umbrales son conservadores para filtrar ruido. Para ver más señales, bajá `PCTL_FUERTE` o `PERSIST_MIN_FUERTE` en `sura_pipeline.py`.

**Error “value is not in available options” en algún selector**: probablemente el Excel no tiene datos en la última fecha para ese ticker. Revisá el archivo de origen.