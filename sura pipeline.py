"""
sura_pipeline.py
================
Pipeline para Request_Sura.xlsx

Arquitectura clave:
- AFP y FFMM se procesan en pipelines TOTALMENTE INDEPENDIENTES.
- _compute_features_afp() y _compute_features_ffmm() son funciones separadas.
- _compute_cross_afp_ffmm() se aplica al final SOLO leyendo outputs, no modifica inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# CONSTANTES
# ============================================================
BLOCK_MARKERS = [
    ("IPSA", "ipsa_shares"),
    ("IPSA - Weights", "ipsa_weight"),
    ("Pension Funds", "afp_mmusd"),
    ("Pension Funds - Weight", "afp_weight"),
    ("Mutual Funds", "ffmm_mmusd"),
    ("Mutual Funds - Weight", "ffmm_weight"),
]

FNEST_EXCLUDE = ["FNEST-030715", "FNEST-040515", "FNEST-090215", "FNEST-150615"]

NOISE_BPS = 0.0005          # 5 bps = 0.0005 en escala decimal
SIGMA_MULT = 0.5            # umbral absoluto adaptativo
PCTL_FUERTE = 0.85          # percentil para Acumulando_Fuerte
Z_EXTREMO = 1.5             # z-score para OW_Extremo / UW_Extremo
PERSIST_MIN_FUERTE = 3      # mínima persistencia para "_Fuerte"


# ============================================================
# 1. LOADER
# ============================================================
def _detect_blocks(raw: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """
    Detecta los 6 bloques de la hoja Weights buscando las strings marcadoras
    en columna 0. Devuelve {key: (header_row, end_row_exclusive)}.
    """
    positions = {}
    rows_col0 = raw.iloc[:, 0].astype(str).values

    for marker, key in BLOCK_MARKERS:
        for i, v in enumerate(rows_col0):
            if str(v).strip() == marker:
                positions[key] = i + 1  # la fila siguiente es el header
                break

    if len(positions) < 6:
        missing = [k for _, k in BLOCK_MARKERS if k not in positions]
        raise ValueError(f"No se detectaron todos los bloques. Faltan: {missing}")

    # Ordenar por fila y calcular fin = inicio del siguiente - 1 (o fin del df)
    keys_ordered = sorted(positions.keys(), key=lambda k: positions[k])
    blocks = {}
    n = len(raw)
    for i, k in enumerate(keys_ordered):
        start = positions[k]
        if i + 1 < len(keys_ordered):
            end = positions[keys_ordered[i + 1]] - 1  # antes del marker siguiente
        else:
            end = n
        blocks[k] = (start, end)
    return blocks


def _parse_block(raw: pd.DataFrame, header_row: int, end_row: int) -> pd.DataFrame:
    """
    Extrae un bloque como DataFrame long con columnas Sector, Ticker, Fecha, Valor.
    """
    header = raw.iloc[header_row].values
    data = raw.iloc[header_row + 1:end_row].copy()
    data.columns = header

    # Detectar columnas de fecha (formato 'YYYY_M')
    date_cols = [c for c in data.columns
                 if isinstance(c, str) and "_" in c
                 and c.split("_")[0].isdigit() and c.split("_")[1].isdigit()]

    # Descartar NaN en Ticker / Sector
    if "Ticker" not in data.columns or "Sector" not in data.columns:
        return pd.DataFrame(columns=["Sector", "Ticker", "Fecha", "Valor"])

    data = data.dropna(subset=["Ticker"])
    data["Ticker"] = data["Ticker"].astype(str).str.strip().str.upper()
    data = data[~data["Ticker"].isin(FNEST_EXCLUDE)]
    data = data[data["Ticker"] != "TICKER"]  # filtra headers duplicados (case-insensitive)

    # Melt a long
    long = data.melt(id_vars=["Sector", "Ticker"], value_vars=date_cols,
                     var_name="YM", value_name="Valor")

    # Parsear fecha YYYY_M a MonthEnd
    def _ym_to_date(s):
        try:
            y, m = s.split("_")
            return pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0)
        except Exception:
            return pd.NaT

    long["Fecha"] = long["YM"].apply(_ym_to_date)
    long = long.drop(columns=["YM"])
    long["Valor"] = pd.to_numeric(long["Valor"], errors="coerce")
    long = long.dropna(subset=["Fecha"])

    # Consolidar duplicados (mismo Ticker en varias filas del Excel — ej: MallPlaza/MALLPLAZA).
    # Se suman los valores porque representan el mismo papel listado dos veces.
    # Se conserva el primer Sector encontrado.
    long = long.groupby(["Ticker", "Fecha"], as_index=False).agg(
        Sector=("Sector", "first"),
        Valor=("Valor", "sum"),
    )

    return long[["Sector", "Ticker", "Fecha", "Valor"]]


def load_sura(xls_source) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.Timestamp]:
    """
    Lee Request_Sura.xlsx y devuelve:
    - blocks_dict: dict con 6 DataFrames long.
    - panel: DataFrame consolidado por (Ticker, Fecha) con pesos y GAPs.
    - last_date: último mes con datos.
    """
    raw = pd.read_excel(xls_source, sheet_name="Weights", header=None, engine="openpyxl")
    block_positions = _detect_blocks(raw)

    blocks = {}
    for key, (start, end) in block_positions.items():
        blocks[key] = _parse_block(raw, start, end)

    # ---- Consolidar en panel ancho ----
    def _pivot(df, name):
        if df.empty:
            return pd.DataFrame()
        return df.rename(columns={"Valor": name})

    ipsa_w = _pivot(blocks["ipsa_weight"], "Peso_IPSA")[["Ticker", "Sector", "Fecha", "Peso_IPSA"]]
    afp_w = _pivot(blocks["afp_weight"], "Peso_AFP")[["Ticker", "Fecha", "Peso_AFP"]]
    ffmm_w = _pivot(blocks["ffmm_weight"], "Peso_FFMM")[["Ticker", "Fecha", "Peso_FFMM"]]
    afp_m = _pivot(blocks["afp_mmusd"], "MMUSD_AFP")[["Ticker", "Fecha", "MMUSD_AFP"]]
    ffmm_m = _pivot(blocks["ffmm_mmusd"], "MMUSD_FFMM")[["Ticker", "Fecha", "MMUSD_FFMM"]]

    # Merge — base es el universo de todos los tickers presentes
    panel = ipsa_w.merge(afp_w, on=["Ticker", "Fecha"], how="outer") \
                  .merge(ffmm_w, on=["Ticker", "Fecha"], how="outer") \
                  .merge(afp_m, on=["Ticker", "Fecha"], how="outer") \
                  .merge(ffmm_m, on=["Ticker", "Fecha"], how="outer")

    # Sector (llenar con la info de ipsa_w o de cualquiera que lo tenga)
    sector_map = pd.concat([
        blocks["ipsa_weight"][["Ticker", "Sector"]],
        blocks["afp_weight"][["Ticker", "Sector"]],
        blocks["ffmm_weight"][["Ticker", "Sector"]]
    ]).dropna().drop_duplicates(subset=["Ticker"])
    sector_dict = dict(zip(sector_map["Ticker"], sector_map["Sector"]))
    panel["Sector"] = panel["Ticker"].map(sector_dict)

    # Rellenar NaN en pesos con 0 (el ticker no estaba en el fondo ese mes)
    for c in ["Peso_IPSA", "Peso_AFP", "Peso_FFMM", "MMUSD_AFP", "MMUSD_FFMM"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)

    # GAPs independientes
    panel["GAP_AFP"] = panel["Peso_AFP"] - panel["Peso_IPSA"]
    panel["GAP_FFMM"] = panel["Peso_FFMM"] - panel["Peso_IPSA"]

    panel = panel.dropna(subset=["Sector", "Fecha"])
    panel = panel.sort_values(["Ticker", "Fecha"]).reset_index(drop=True)

    # Universo final: tickers con IPSA > 0 o fondo > 0 en la última fecha
    last_date = panel["Fecha"].max()
    last_snap = panel[panel["Fecha"] == last_date]
    valid_tickers = last_snap[
        (last_snap["Peso_IPSA"] > 0) |
        (last_snap["Peso_AFP"] > 0) |
        (last_snap["Peso_FFMM"] > 0)
    ]["Ticker"].unique()
    panel = panel[panel["Ticker"].isin(valid_tickers)].copy()

    return blocks, panel, last_date


# ============================================================
# 2. FEATURES POR UNIVERSO — FUNCIONES SEPARADAS
# ============================================================
def _compute_features_one_universe(panel: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Calcula todas las features para UN universo (AFP o FFMM).
    suffix: 'AFP' o 'FFMM' → lee GAP_AFP o GAP_FFMM y genera columnas con ese sufijo.
    """
    df = panel.copy().sort_values(["Ticker", "Fecha"])
    gap_col = f"GAP_{suffix}"
    g = df.groupby("Ticker", group_keys=False)

    df[f"Delta_GAP_{suffix}"] = g[gap_col].diff()
    df[f"Cambio_Flujo_{suffix}"] = g[f"Delta_GAP_{suffix}"].diff()
    df[f"Delta_GAP_MA3_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    df[f"Delta_GAP_MA6_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.rolling(6, min_periods=3).mean())
    df[f"Fuerza_Flujo_{suffix}"] = df[f"Delta_GAP_{suffix}"] - df[f"Delta_GAP_MA3_{suffix}"]
    df[f"GAP_MA6_{suffix}"] = g[gap_col].transform(lambda s: s.rolling(6, min_periods=3).mean())
    df[f"GAP_STD6_{suffix}"] = g[gap_col].transform(lambda s: s.rolling(6, min_periods=3).std())
    df[f"GAP_Z6_{suffix}"] = (df[gap_col] - df[f"GAP_MA6_{suffix}"]) / df[f"GAP_STD6_{suffix}"].replace(0, np.nan)
    df[f"GAP_Pctl_{suffix}"] = g[gap_col].transform(lambda s: s.rank(pct=True))
    df[f"Delta_Pctl_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.abs().rank(pct=True))
    df[f"Delta_GAP_3M_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df[f"Delta_GAP_6M_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    df[f"Sigma_Delta_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(lambda s: s.std())

    # Persistencia: # meses consecutivos con mismo signo de Delta_GAP
    def _persistencia(s):
        signs = np.sign(s.fillna(0)).values
        out = np.zeros(len(signs), dtype=int)
        for i in range(len(signs)):
            if signs[i] == 0:
                out[i] = 0
            elif i == 0:
                out[i] = int(signs[i])
            elif signs[i] == signs[i - 1]:
                out[i] = out[i - 1] + int(signs[i])
            else:
                out[i] = int(signs[i])
        return pd.Series(out, index=s.index)

    df[f"Persistencia_{suffix}"] = g[f"Delta_GAP_{suffix}"].transform(_persistencia)

    return df


def _classify_one_universe(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Aplica el sistema de etiquetas (Posicionamiento, Dirección, Señal) a un universo.
    """
    df = df.copy()
    gap = df[f"GAP_{suffix}"]
    delta = df[f"Delta_GAP_{suffix}"]
    z = df[f"GAP_Z6_{suffix}"]
    pctl = df[f"Delta_Pctl_{suffix}"]
    sigma = df[f"Sigma_Delta_{suffix}"]
    persist = df[f"Persistencia_{suffix}"]

    # ---- Posicionamiento ----
    # Regla: primero se define el signo (OW vs UW vs Neutral) a partir del GAP.
    # Luego el z-score solo decide si el papel está en EXTREMO DENTRO de su propio signo.
    # El z-score nunca voltea el signo.
    def _pos(row_gap, row_z):
        if pd.isna(row_gap):
            return "Sin dato"
        if abs(row_gap) < NOISE_BPS:
            return "Neutral"
        if row_gap > 0:
            # OW: extremo si el z es alto (aún más OW que su promedio)
            if pd.notna(row_z) and row_z >= Z_EXTREMO:
                return "OW_Extremo"
            return "OW"
        else:
            # UW: extremo si el z es muy bajo (aún más UW que su promedio)
            if pd.notna(row_z) and row_z <= -Z_EXTREMO:
                return "UW_Extremo"
            return "UW"

    df[f"Posicionamiento_{suffix}"] = [
        _pos(g, zv) for g, zv in zip(gap, z)
    ]

    # ---- Umbral adaptativo: ¿el movimiento es "significativo"? ----
    thr_abs = np.maximum(NOISE_BPS, SIGMA_MULT * sigma.fillna(NOISE_BPS))
    significativo = (delta.abs() > thr_abs) & (pctl >= PCTL_FUERTE)

    # ---- Dirección ----
    def _dir(d, persist_val, signif):
        if pd.isna(d):
            return "Sin dato"
        if abs(d) < NOISE_BPS:
            return "Plano"
        if d > 0:
            if signif and persist_val >= PERSIST_MIN_FUERTE:
                return "Acumulando_Fuerte"
            return "Acumulando"
        else:
            if signif and persist_val <= -PERSIST_MIN_FUERTE:
                return "Reduciendo_Fuerte"
            return "Reduciendo"

    df[f"Direccion_{suffix}"] = [
        _dir(d, p, s) for d, p, s in zip(delta, persist, significativo)
    ]

    # ---- Señal (matriz explícita) ----
    SIGNAL_MATRIX = {
        ("UW_Extremo", "Acumulando_Fuerte"): "BUY_FUERTE",
        ("UW", "Acumulando_Fuerte"): "BUY",
        ("UW_Extremo", "Acumulando"): "BUY",
        ("UW", "Acumulando"): "BUY_LIGHT",
        ("Neutral", "Acumulando_Fuerte"): "BUY_LIGHT",
        ("OW", "Acumulando_Fuerte"): "BUY_LIGHT",
        ("OW", "Acumulando"): "HOLD",
        ("OW_Extremo", "Acumulando"): "HOLD",
        ("OW_Extremo", "Reduciendo_Fuerte"): "SELL",
        ("OW", "Reduciendo_Fuerte"): "SELL",
        ("OW_Extremo", "Reduciendo"): "SELL_LIGHT",
        ("Neutral", "Reduciendo_Fuerte"): "SELL_LIGHT",
        ("UW", "Reduciendo_Fuerte"): "SELL_FUERTE",
        ("UW_Extremo", "Reduciendo_Fuerte"): "SELL_FUERTE",
    }

    def _signal(pos, direc):
        key = (pos, direc)
        if key in SIGNAL_MATRIX:
            return SIGNAL_MATRIX[key]
        if direc == "Plano" or pos == "Sin dato" or direc == "Sin dato":
            return "HOLD"
        return "HOLD"

    df[f"Senal_{suffix}"] = [
        _signal(p, d) for p, d in zip(df[f"Posicionamiento_{suffix}"], df[f"Direccion_{suffix}"])
    ]

    # ---- Semáforo derivado ----
    def _sem(sig):
        if sig in ("BUY_FUERTE", "BUY"):
            return "🟢"
        if sig in ("BUY_LIGHT",):
            return "🟢"
        if sig in ("SELL_FUERTE", "SELL"):
            return "🔴"
        if sig in ("SELL_LIGHT",):
            return "🔴"
        return "🟡"

    df[f"Sem_{suffix}"] = df[f"Senal_{suffix}"].apply(_sem)

    # ---- FlowScore 0-100 ----
    raw_score = (
        35 * (df[f"Delta_GAP_{suffix}"] > 0).astype(int) +
        25 * (df[f"Cambio_Flujo_{suffix}"] > 0).astype(int) +
        20 * (df[f"Fuerza_Flujo_{suffix}"] > 0).astype(int) +
        20 * (df[f"GAP_{suffix}"] > 0).astype(int)
    )
    df[f"FlowScore_{suffix}"] = raw_score.astype(float)

    return df


def _compute_features_afp(panel: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo AFP — independiente de FFMM."""
    df = _compute_features_one_universe(panel, "AFP")
    df = _classify_one_universe(df, "AFP")
    return df


def _compute_features_ffmm(panel: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo FFMM — independiente de AFP."""
    df = _compute_features_one_universe(panel, "FFMM")
    df = _classify_one_universe(df, "FFMM")
    return df


# ============================================================
# 3. CRUCE AFP ↔ FFMM (solo lee outputs, no modifica)
# ============================================================
def _compute_cross_afp_ffmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas cruzadas DESPUÉS de que ambos pipelines corrieron por separado.
    No modifica columnas AFP ni FFMM existentes.
    """
    df = df.copy().sort_values(["Ticker", "Fecha"])
    g = df.groupby("Ticker", group_keys=False)

    df["Divergencia_GAP"] = df["GAP_AFP"] - df["GAP_FFMM"]
    df["Divergencia_Z6"] = g["Divergencia_GAP"].transform(
        lambda s: (s - s.rolling(6, min_periods=3).mean()) /
                  s.rolling(6, min_periods=3).std().replace(0, np.nan)
    )

    # Correlación rolling 6M entre delta_AFP y delta_FFMM
    def _rolling_corr(sub):
        a = sub["Delta_GAP_AFP"]
        b = sub["Delta_GAP_FFMM"]
        return a.rolling(6, min_periods=4).corr(b)

    df["Corr_6M"] = g[["Delta_GAP_AFP", "Delta_GAP_FFMM"]].apply(
        lambda sub: _rolling_corr(sub)
    ).reset_index(level=0, drop=True)

    # Lead_Lag: por ticker, lag que maximiza correlación cruzada (últimos 24 puntos)
    def _lead_lag_per_ticker(sub, max_lag=3, window=24):
        a = sub["Delta_GAP_AFP"].dropna().tail(window).values
        b = sub["Delta_GAP_FFMM"].dropna().tail(window).values
        if len(a) < 8 or len(b) < 8:
            return np.nan
        n = min(len(a), len(b))
        a = a[-n:]
        b = b[-n:]
        best_lag, best_corr = 0, -2
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x, y = a[:lag], b[-lag:]
            elif lag > 0:
                x, y = a[lag:], b[:-lag]
            else:
                x, y = a, b
            if len(x) < 4:
                continue
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            c = np.corrcoef(x, y)[0, 1]
            if c > best_corr:
                best_corr = c
                best_lag = lag
        return best_lag

    ll_map = df.groupby("Ticker").apply(lambda sub: _lead_lag_per_ticker(sub)).to_dict()
    df["Lead_Lag"] = df["Ticker"].map(ll_map)

    # Liderazgo del mes (solo tiene sentido en última fecha)
    def _liderazgo(row, df_ticker):
        t = row["Ticker"]
        fecha = row["Fecha"]
        sub = df_ticker.get(t)
        if sub is None:
            return "Sin_Señal"
        row_idx = sub.index[sub["Fecha"] == fecha]
        if len(row_idx) == 0:
            return "Sin_Señal"
        i = row_idx[0]
        pos_in_sub = sub.index.get_loc(i)

        d_afp = row["Delta_GAP_AFP"]
        d_ffmm = row["Delta_GAP_FFMM"]

        if pd.isna(d_afp) or pd.isna(d_ffmm):
            return "Sin_Señal"
        if abs(d_afp) < NOISE_BPS and abs(d_ffmm) < NOISE_BPS:
            return "Sin_Señal"

        # Consenso / divergencia sincrónicos
        if d_afp > 0 and d_ffmm > 0:
            return "Consenso_Compra"
        if d_afp < 0 and d_ffmm < 0:
            return "Consenso_Venta"
        if d_afp * d_ffmm < 0:
            return "Divergencia_Flujos"

        # Lead check: AFP se movió, FFMM plano, FFMM se mueve 1-2 meses después
        future_idx = list(range(pos_in_sub + 1, min(pos_in_sub + 3, len(sub))))
        if d_afp > 0 and abs(d_ffmm) < NOISE_BPS:
            for fi in future_idx:
                if sub["Delta_GAP_FFMM"].iloc[fi] > 0:
                    return "Lidera_AFP"
        if d_ffmm > 0 and abs(d_afp) < NOISE_BPS:
            for fi in future_idx:
                if sub["Delta_GAP_AFP"].iloc[fi] > 0:
                    return "Lidera_FFMM"

        return "Sin_Señal"

    df_ticker = {t: sub for t, sub in df.groupby("Ticker")}
    df["Liderazgo_del_mes"] = df.apply(lambda r: _liderazgo(r, df_ticker), axis=1)

    return df


# ============================================================
# 4. EVENTS
# ============================================================
def build_events(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Events log por universo: fechas donde cambia Posicionamiento, Dirección o Señal."""
    events = []
    for tk, sub in df.groupby("Ticker"):
        sub = sub.sort_values("Fecha")
        prev_p, prev_d, prev_s = None, None, None
        for _, r in sub.iterrows():
            p = r.get(f"Posicionamiento_{suffix}")
            d = r.get(f"Direccion_{suffix}")
            s = r.get(f"Senal_{suffix}")
            if (p != prev_p) or (d != prev_d) or (s != prev_s):
                events.append({
                    "Ticker": tk,
                    "Fecha": r["Fecha"],
                    "Universo": suffix,
                    "Posicionamiento": p,
                    "Direccion": d,
                    "Senal": s,
                    "GAP": r.get(f"GAP_{suffix}"),
                    "Delta_GAP": r.get(f"Delta_GAP_{suffix}"),
                    "Persistencia": r.get(f"Persistencia_{suffix}")
                })
                prev_p, prev_d, prev_s = p, d, s
    return pd.DataFrame(events)


# ============================================================
# 5. BUILDER PRINCIPAL
# ============================================================
def build_outputs(xls_source):
    """
    Orquesta todo el pipeline:
    1. Carga el Excel.
    2. Corre AFP y FFMM en pipelines separados.
    3. Calcula métricas de cruce.
    4. Genera event logs por universo.
    """
    blocks, panel, last_date = load_sura(xls_source)

    # Pipelines independientes
    df_afp = _compute_features_afp(panel)
    df_ffmm_only = _compute_features_ffmm(panel)

    # Merge: mantener columnas AFP de df_afp y columnas FFMM de df_ffmm_only
    ffmm_cols = [c for c in df_ffmm_only.columns if c.endswith("_FFMM")]
    base_cols = ["Ticker", "Fecha"]
    df = df_afp.merge(
        df_ffmm_only[base_cols + ffmm_cols],
        on=base_cols,
        how="left",
        suffixes=("", "_dup")
    )
    # Eliminar duplicados
    for c in df.columns:
        if c.endswith("_dup"):
            df = df.drop(columns=[c])

    # Capa cruzada (solo lee, no modifica)
    df = _compute_cross_afp_ffmm(df)

    # Snapshot última fecha
    snap_last = df[df["Fecha"] == last_date].copy()

    # Events separados por universo
    events_afp = build_events(df, "AFP")
    events_ffmm = build_events(df, "FFMM")

    return {
        "panel": panel,
        "df": df,
        "snap_last": snap_last,
        "events_afp": events_afp,
        "events_ffmm": events_ffmm,
        "last_date": last_date,
        "blocks": blocks,
    }
